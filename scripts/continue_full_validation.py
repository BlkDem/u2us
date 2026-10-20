"""Continue full_validation RU->EN in the same protocol; never reads test."""
import hashlib
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
import torch.nn.functional as F
import yaml
from micronmt.model import MicroTransformer
from micronmt.sp_tokenizer import SentencePieceTokenizer
from scripts.diagnose_training import batch, evaluate, rows_write, sha, dump

BASE_RUN = ROOT / "runs" / "full_validation"
BASE_REPORT = ROOT / "reports" / "full_validation"
NEW_RUN = ROOT / "runs" / "full_validation_continued"
NEW_REPORT = ROOT / "reports" / "full_validation_continued"

ADDITIONAL_STEPS = 1500


def main():
    torch.set_num_threads(2)
    started = time.perf_counter()
    if NEW_RUN.exists() or NEW_REPORT.exists():
        raise SystemExit("Refusing existing continued output directories")
    NEW_RUN.mkdir()
    NEW_REPORT.mkdir()

    tokpath = ROOT / "data/tokenizer/real_baseline/shared.model"
    tok = SentencePieceTokenizer(tokpath)
    paths = {s: ROOT / f"data/real_baseline/ru_en/{s}.jsonl" for s in ("train", "valid")}
    rows = {s: [json.loads(l) for l in p.read_text().splitlines()] for s, p in paths.items()}
    assert len(rows["train"]) == 8000

    max_len = 192
    cohort_files = {
        "diagnostic_short100": BASE_REPORT / "diagnostic_short100.jsonl",
        "full_first100": BASE_REPORT / "full_first100.jsonl",
    }
    cohorts = {}
    for name, path in cohort_files.items():
        if path.exists():
            cohorts[name] = [json.loads(l) for l in path.read_text().splitlines()]
        else:
            cohorts[name] = rows["valid"][:100]

    generator = torch.Generator().manual_seed(42)
    all_batches = []
    while len(all_batches) < 1500 + ADDITIONAL_STEPS:
        order = torch.randperm(len(rows["train"]), generator=generator).tolist()
        all_batches.extend(order[i : i + 32] for i in range(0, len(order), 32))
    all_batches = all_batches[: 1500 + ADDITIONAL_STEPS]
    new_batches = all_batches[1500:]
    assert len(new_batches) == ADDITIONAL_STEPS
    dump(NEW_REPORT / "batch_indices.json", new_batches)

    config_path = ROOT / "configs" / "full_validation_ru_en.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)
    config["max_len"] = max_len
    config["init_mode"] = config.get("init_mode", "controlled")

    saved = torch.load(BASE_RUN / "last.pt", map_location="cpu", weights_only=True)
    model = MicroTransformer(tok.vocab_size, **{k: config[k] for k in ("d_model", "nhead", "encoder_layers", "decoder_layers", "dim_feedforward", "dropout", "max_len")}, init_mode=config.get("init_mode", "controlled"))
    model.load_state_dict(saved["model"])

    opt = torch.optim.AdamW(model.parameters(), lr=config["lr"])
    if "optimizer" in saved:
        opt.load_state_dict(saved["optimizer"])
    for group in opt.param_groups:
        group["lr"] = config["lr"]

    if "python_rng_state" in saved:
        random.setstate(saved["python_rng_state"])
    if "torch_rng_state" in saved:
        torch.set_rng_state(saved["torch_rng_state"])

    base_best_chrF = None
    base_summary_path = BASE_REPORT / "summary.json"
    if base_summary_path.exists():
        base_summary = json.loads(base_summary_path.read_text())
        base_best_chrF = base_summary.get("best", {}).get("diagnostic_short100", {}).get("chrF")
    if base_best_chrF is None:
        base_best_chrF = -1.0
    best = base_best_chrF
    selected_step = 1500
    global_step = 1500

    from micronmt.checkpoint import tokenizer_metadata

    torch.manual_seed(42)
    torch.save(model.state_dict(), NEW_RUN / "initial.pt")

    history = []
    start_entry = dict(step=global_step, elapsed_seconds=time.perf_counter() - started, cohorts={n: evaluate(model, rs, tok, max_len) for n, rs in cohorts.items()})
    history.append(start_entry)
    rows_write(NEW_REPORT / "history.jsonl", history)
    payload = dict(model=model.state_dict(), config=config, tokenizer=tokenizer_metadata(tok), direction="ru2en", step=global_step)
    torch.save(payload, NEW_RUN / f"step_{global_step:04d}.pt")
    if start_entry["cohorts"]["diagnostic_short100"]["chrF"] > best:
        best = start_entry["cohorts"]["diagnostic_short100"]["chrF"]
        selected_step = global_step
        torch.save(payload, NEW_RUN / "best.pt")
    print(global_step, "chrF", start_entry["cohorts"]["diagnostic_short100"]["chrF"], "full100", start_entry["cohorts"]["full_first100"]["chrF"], flush=True)

    for indices in new_batches:
        model.train()
        src, tgt = batch([rows["train"][i] for i in indices], tok, max_len)
        opt.zero_grad()
        logits = model(src, tgt[:, :-1])
        loss = F.cross_entropy(logits.flatten(0, 1), tgt[:, 1:].flatten(), ignore_index=0, label_smoothing=config["label_smoothing"])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        global_step += 1

        if global_step % 250 == 0:
            entry = dict(step=global_step, elapsed_seconds=time.perf_counter() - started, cohorts={n: evaluate(model, rs, tok, max_len) for n, rs in cohorts.items()})
            history.append(entry)
            rows_write(NEW_REPORT / "history.jsonl", history)
            payload = dict(model=model.state_dict(), config=config, tokenizer=tokenizer_metadata(tok), direction="ru2en", step=global_step)
            torch.save(payload, NEW_RUN / f"step_{global_step:04d}.pt")
            chrF = entry["cohorts"]["diagnostic_short100"]["chrF"]
            if chrF > best:
                best = chrF
                selected_step = global_step
                torch.save(payload, NEW_RUN / "best.pt")
            print(global_step, "chrF", chrF, "full100", entry["cohorts"]["full_first100"]["chrF"], flush=True)

    final_payload = dict(model=model.state_dict(), config=config, tokenizer=tokenizer_metadata(tok), direction="ru2en", step=global_step, optimizer=opt.state_dict(), torch_rng_state=torch.get_rng_state(), python_rng_state=random.getstate())
    torch.save(final_payload, NEW_RUN / "last.pt")

    selected_entry = None
    for e in history:
        if e["step"] == selected_step:
            selected_entry = e
    summary = dict(
        status="completed",
        production_approved=False,
        license="unknown",
        base_run=str(BASE_RUN),
        config=config,
        training=dict(batch_size=32, seed=42, additional_steps=ADDITIONAL_STEPS, total_steps=global_step, lr=config["lr"], label_smoothing=config["label_smoothing"], scheduler="constant", optimizer="AdamW defaults weight_decay=.01", gradient_clip=1.0, eval_every=250),
        data=dict(train_rows=len(rows["train"]), valid_rows=len(rows["valid"])),
        tokenizer_sha256=sha(tokpath),
        batch_sha256=sha(NEW_REPORT / "batch_indices.json"),
        cohort_hashes={n: sha(BASE_REPORT / f"{n}.jsonl") if (BASE_REPORT / f"{n}.jsonl").exists() else None for n in cohorts},
        selected_step=selected_step,
        best={n: {k: v for k, v in m.items() if k != "samples"} for n, m in selected_entry["cohorts"].items()} if selected_entry else None,
        elapsed_seconds=time.perf_counter() - started,
        best_sha256=sha(NEW_RUN / "best.pt"),
        limitations=["Continuation of full_validation; no test accessed.", "Single seed, license unknown, production not approved."],
    )
    dump(NEW_REPORT / "summary.json", summary)
    print("done", global_step, "best_step", selected_step, "best_chrF", best, flush=True)


if __name__ == "__main__":
    main()
