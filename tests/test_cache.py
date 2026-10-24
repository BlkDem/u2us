import torch

from micronmt.model import MicroTransformer


def build_model():
    torch.manual_seed(0)
    return MicroTransformer(64, d_model=16, nhead=2, encoder_layers=1, decoder_layers=1,
                            dim_feedforward=32, dropout=0.0, max_len=16).eval()


def test_cached_greedy_matches_greedy():
    model = build_model()
    src = torch.randint(4, 64, (2, 7))
    src[1, 5:] = 0
    expected = model.greedy(src, max_len=12)
    actual = model.greedy_cached(src, max_len=12)
    assert torch.equal(expected, actual)


def test_cached_greedy_respects_padding_and_max_len():
    model = build_model()
    src = torch.randint(4, 64, (3, 9))
    src[0, 6:] = 0
    src[2, 8:] = 0
    out = model.greedy_cached(src, max_len=10)
    assert out.size(0) == 3
    assert out.size(1) <= 10
    assert torch.equal(out[:, 0], torch.full((3,), 2))
