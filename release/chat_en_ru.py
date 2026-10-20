import subprocess, sys, os

MODEL_DIR = os.path.join(os.path.dirname(__file__), "en_ru")
while True:
    try:
        text = input("en> ").strip()
    except (EOFError, KeyboardInterrupt):
        break
    if not text:
        continue
    r = subprocess.run(
        [sys.executable, "-m", "micronmt.onnx_translate",
         "--model-dir", MODEL_DIR, "--text", text, "--source", "en"],
        capture_output=True, text=True,
    )
    print("ru>", r.stdout.strip() if r.stdout.strip() else r.stderr.strip())
