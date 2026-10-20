import subprocess, sys, os

MODEL_DIR = os.path.join(os.path.dirname(__file__), "ru_en")
while True:
    try:
        text = input("ru> ").strip()
    except (EOFError, KeyboardInterrupt):
        break
    if not text:
        continue
    r = subprocess.run(
        [sys.executable, "-m", "micronmt.onnx_translate",
         "--model-dir", MODEL_DIR, "--text", text, "--source", "ru"],
        capture_output=True, text=True,
    )
    print("en>", r.stdout.strip() if r.stdout.strip() else r.stderr.strip())
