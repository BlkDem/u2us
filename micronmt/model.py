import math
import torch
import torch.nn.functional as F
from torch import nn


class MicroTransformer(nn.Module):
    """Small seq2seq Transformer. Direction/context are represented in source tokens."""
    def __init__(self, vocab_size, d_model=128, nhead=4, encoder_layers=2,
                 decoder_layers=2, dim_feedforward=512, dropout=.1, max_len=256,
                 init_mode='controlled'):
        super().__init__()
        self.d_model, self.max_len = d_model, max_len
        self.nhead = nhead
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos = nn.Embedding(max_len, d_model)
        layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout,
                                           batch_first=True, norm_first=True)
        dec = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout,
                                         batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, encoder_layers)
        self.decoder = nn.TransformerDecoder(dec, decoder_layers)
        self.norm = nn.LayerNorm(d_model)
        self.output = nn.Linear(d_model, vocab_size, bias=False)
        self.output.weight = self.embedding.weight
        if init_mode not in ('controlled', 'legacy'):
            raise ValueError('init_mode must be controlled or legacy')
        if init_mode == 'controlled':
            with torch.random.fork_rng():
                torch.manual_seed(424242)
                with torch.no_grad():
                    self.embedding.weight.normal_(0, d_model ** -.5)
                    self.embedding.weight[0].zero_()

    def add_pos(self, x):
        p = torch.arange(x.size(1), device=x.device).unsqueeze(0)
        return x + self.pos(p.clamp(max=self.max_len - 1))

    def forward(self, src, tgt, pad_id=0):
        src_key = src.eq(pad_id)
        tgt_key = tgt.eq(pad_id)
        memory = self.encoder(self.add_pos(self.embedding(src) * math.sqrt(self.d_model)),
                              src_key_padding_mask=src_key)
        causal = torch.zeros(tgt.size(1), tgt.size(1), device=tgt.device).masked_fill(
            torch.triu(torch.ones(tgt.size(1), tgt.size(1), device=tgt.device, dtype=torch.bool), diagonal=1), float('-inf'))
        out = self.decoder(self.add_pos(self.embedding(tgt) * math.sqrt(self.d_model)), memory,
                           tgt_mask=causal, tgt_is_causal=True, tgt_key_padding_mask=tgt_key,
                           memory_key_padding_mask=src_key)
        return self.output(self.norm(out))

    @torch.no_grad()
    def greedy(self, src, bos_id=2, eos_id=3, max_len=256, pad_id=0):
        out = torch.full((src.size(0), 1), bos_id, dtype=torch.long, device=src.device)
        finished = torch.zeros(src.size(0), dtype=torch.bool, device=src.device)
        for _ in range(max_len - 1):
            nxt = self(src, out, pad_id).argmax(-1)[:, -1:]
            nxt = torch.where(finished[:, None], torch.full_like(nxt, eos_id), nxt)
            out = torch.cat([out, nxt], 1)
            finished |= nxt[:, 0].eq(eos_id)
            if finished.all(): break
        return out

    def _project_qkv(self, attention, x):
        d = attention.embed_dim
        weight, bias = attention.in_proj_weight, attention.in_proj_bias
        q = F.linear(x, weight[:d], bias[:d] if bias is not None else None)
        k = F.linear(x, weight[d:2 * d], bias[d:2 * d] if bias is not None else None)
        v = F.linear(x, weight[2 * d:], bias[2 * d:] if bias is not None else None)
        return q, k, v

    def _cached_decoder_step(self, x, memory, src_key, cache, cross_cache, cross_mask):
        batch = x.size(0)
        for index, layer in enumerate(self.decoder.layers):
            attention = layer.self_attn
            q, k, v = self._project_qkv(attention, layer.norm1(x))
            heads = attention.num_heads
            head_dim = attention.head_dim
            q = q.view(batch, 1, heads, head_dim).transpose(1, 2)
            k = k.view(batch, 1, heads, head_dim).transpose(1, 2)
            v = v.view(batch, 1, heads, head_dim).transpose(1, 2)
            past_k, past_v = cache[index]
            if past_k is not None:
                k = torch.cat([past_k, k], dim=2)
                v = torch.cat([past_v, v], dim=2)
            cache[index] = (k, v)
            attn = F.scaled_dot_product_attention(q, k, v)
            attn = attention.out_proj(attn.transpose(1, 2).reshape(batch, 1, attention.embed_dim))
            x = x + layer.dropout1(attn)
            cross = layer.multihead_attn
            if cross_cache[index] is None:
                _, cross_k, cross_v = self._project_qkv(cross, memory)
                cross_cache[index] = (
                    cross_k.view(batch, memory.size(1), cross.num_heads, cross.head_dim).transpose(1, 2),
                    cross_v.view(batch, memory.size(1), cross.num_heads, cross.head_dim).transpose(1, 2),
                )
            cross_k, cross_v = cross_cache[index]
            cross_q, _, _ = self._project_qkv(cross, layer.norm2(x))
            cross_q = cross_q.view(batch, 1, cross.num_heads, cross.head_dim).transpose(1, 2)
            cross_attn = F.scaled_dot_product_attention(cross_q, cross_k, cross_v, attn_mask=cross_mask)
            cross_attn = cross.out_proj(cross_attn.transpose(1, 2).reshape(batch, 1, cross.embed_dim))
            x = x + layer.dropout2(cross_attn)
            x = x + layer.dropout3(layer.linear2(layer.dropout(layer.activation(layer.linear1(layer.norm3(x))))))
        return self.output(self.norm(x))

    @torch.no_grad()
    def greedy_cached(self, src, bos_id=2, eos_id=3, max_len=256, pad_id=0):
        device = src.device
        src_key = src.eq(pad_id)
        memory = self.encoder(self.add_pos(self.embedding(src) * math.sqrt(self.d_model)),
                              src_key_padding_mask=src_key)
        out = torch.full((src.size(0), 1), bos_id, dtype=torch.long, device=device)
        finished = torch.zeros(src.size(0), dtype=torch.bool, device=device)
        cache = [(None, None) for _ in self.decoder.layers]
        cross_cache = [None for _ in self.decoder.layers]
        cross_mask = torch.zeros(src.size(0), 1, 1, src.size(1), device=device).masked_fill(
            src_key[:, None, None, :], float('-inf'))
        position = torch.zeros(1, dtype=torch.long, device=device)
        logits = self._cached_decoder_step(
            self.embedding(out) * math.sqrt(self.d_model) + self.pos(position), memory, src_key,
            cache, cross_cache, cross_mask)
        for step in range(max_len - 1):
            nxt = logits.argmax(-1)
            nxt = torch.where(finished[:, None], torch.full_like(nxt, eos_id), nxt)
            out = torch.cat([out, nxt], 1)
            finished |= nxt[:, 0].eq(eos_id)
            if finished.all():
                break
            position = torch.full((1,), min(step + 1, self.max_len - 1), dtype=torch.long, device=device)
            logits = self._cached_decoder_step(
                self.embedding(nxt) * math.sqrt(self.d_model) + self.pos(position), memory, src_key,
                cache, cross_cache, cross_mask)
        return out

