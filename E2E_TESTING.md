# E2E-тестирование модели

Этот документ описывает полностью локальный запуск тестового агента. Внешние API и облачные сервисы не используются.

## 1. Подготовка окружения

Из корня проекта:

```bash
python -m pip install --break-system-packages --user -e ".[train]"
```

Если PyTorch ещё не установлен, для CPU:

```bash
python -m pip install --break-system-packages --user torch \
  --index-url https://download.pytorch.org/whl/cpu
```

Проверка:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

## 2. Получение checkpoint

Тестировать можно любой checkpoint, совместимый с проектом:

```text
runs/tiny/best.pt
runs/small/best.pt
artifacts/ru_en/model.pt
```

Если checkpoint ещё не создан, сделать smoke model:

```bash
python scripts/make_toy_data.py --out /tmp/u2us-toy/raw
python scripts/prepare_data.py \
  --input /tmp/u2us-toy/raw/train.jsonl \
  --out /tmp/u2us-toy/data \
  --valid-ratio 0.25 \
  --test-ratio 0.25

cp configs/tiny.yaml /tmp/u2us-toy/tiny.yaml
sed -i 's/epochs: 100/epochs: 1/; s#runs/tiny#/tmp/u2us-toy/run#' /tmp/u2us-toy/tiny.yaml
python -m micronmt.train \
  --config /tmp/u2us-toy/tiny.yaml \
  --data /tmp/u2us-toy/data/train.jsonl
```

Для настоящего теста качества нужен checkpoint, обученный на реальном лицензированном RU↔EN корпусе. Toy checkpoint проверяет только работоспособность pipeline.

## 3. Быстрый функциональный агент

Запуск встроенных тестовых случаев:

```bash
python -m micronmt.test_agent \
  --checkpoint runs/tiny/best.pt \
  --out reports/agent.json
```

Агент проверяет:

- RU → EN;
- EN → RU;
- context input;
- protected identifiers;
- отсутствие исключений;
- наличие ожидаемого результата;
- latency каждого запроса.

Программа возвращает exit code `0`, только если все тесты прошли.

Для quality decoding:

```bash
python -m micronmt.test_agent \
  --checkpoint runs/tiny/best.pt \
  --beam-size 4 \
  --out reports/agent-beam4.json
```

## 4. Полный E2E-тест с английскими текстами

Команда ниже автоматически берёт набор английских предложений, прогоняет их через локальную модель EN → RU и сохраняет все переводы:

```bash
python scripts/e2e_test.py \
  --checkpoint runs/tiny/best.pt \
  --output reports/e2e.jsonl \
  --report reports/e2e_report.json
```

Скрипт генерирует/использует следующие типы текстов:

- deployment и IT;
- hotel/stay ambiguity;
- conditional sentences;
- long-context semantics;
- API identifiers;
- phrasal verbs;
- legal language;
- passive/reported speech;
- offline/privacy requirements.

Пример stdout:

```json
{
  "checkpoint": "runs/tiny/best.pt",
  "total": 10,
  "successful": 10,
  "success_rate": 1.0,
  "avg_latency_ms": 420.1,
  "output": "reports/e2e.jsonl"
}
```

`successful` означает, что inference завершился без exception и вернул непустую строку. Это не означает, что перевод семантически правильный. Семантическое качество оценивается отдельно человеком, BLEU/chrF и COMET.

## 5. Просмотр переводов

```bash
cat reports/e2e.jsonl
```

Или красиво:

```bash
python - <<'PY'
import json
for line in open('reports/e2e.jsonl', encoding='utf8'):
    x=json.loads(line)
    print(f"[{x['id']}] EN: {x['source']}\n    RU: {x['translation']}\n    latency: {x['latency_ms']:.1f} ms\n")
PY
```

## 6. Собственные английские тексты

Создать файл:

```text
my_english.txt
```

По одному предложению на строку:

```text
The service must remain available during the upgrade.
We need to preserve the existing API endpoint.
The contract does not allow the supplier to change the terms unilaterally.
```

Запуск:

```bash
python scripts/e2e_test.py \
  --checkpoint runs/tiny/best.pt \
  --texts-file my_english.txt \
  --output reports/custom-e2e.jsonl \
  --report reports/custom-e2e.json
```

## 7. Тестирование reverse direction

E2E script по умолчанию тестирует EN → RU. Для RU → EN используйте test agent с custom cases либо Python API:

```python
from micronmt.api import Translator

translator = Translator('runs/tiny/best.pt')
result = translator.translate(
    'Я решил остаться в отеле ещё на две ночи.',
    source='ru',
    target='en',
    beam_size=4,
)
print(result['translation'])
```

## 8. Терминология и контекст

```bash
cat > terminology.json <<'JSON'
{
  "deployment": ["развертывание", "деплой"],
  "pipeline": ["конвейер", "pipeline"]
}
JSON

python -m micronmt.infer \
  --checkpoint runs/tiny/best.pt \
  --text "The deployment pipeline failed." \
  --source en \
  --target ru \
  --context "The application was updated yesterday." \
  --terms terminology.json \
  --beam-size 4 \
  --json
```

## 9. Проверка ONNX E2E

Сначала экспортировать split runtime:

```bash
python -m micronmt.export_seq2seq \
  --checkpoint runs/tiny/best.pt \
  --out-dir artifacts/tiny-onnx \
  --sequence-len 128
```

Затем запустить autoregressive ONNX inference:

```bash
python -m micronmt.onnx_translate \
  --model-dir artifacts/tiny-onnx \
  --checkpoint runs/tiny/best.pt \
  --text "The service must remain available during the upgrade." \
  --source en \
  --sequence-len 128
```

## 10. Как интерпретировать результат

Нельзя считать модель качественной только потому, что:

```text
success_rate = 1.0
```

Это означает только техническую работоспособность inference.

Для оценки качества запускайте:

```bash
python -m micronmt.predict \
  --checkpoint runs/tiny/best.pt \
  --input data/ru_en/test.jsonl \
  --output reports/predictions.jsonl

python -m micronmt.evaluate \
  --reference data/ru_en/test.jsonl \
  --hypothesis reports/predictions.jsonl \
  --out reports/evaluation.json
```

Также необходимо вручную проверить минимум:

- adequacy;
- omissions/additions;
- grammar;
- naturalness;
- terminology;
- idioms;
- ambiguity;
- protected identifiers.

## 11. Запуск перед release

```bash
set -e
python -m py_compile micronmt/*.py scripts/*.py
python -m micronmt.test_agent --checkpoint runs/tiny/best.pt --out reports/agent.json
python scripts/e2e_test.py --checkpoint runs/tiny/best.pt --output reports/e2e.jsonl --report reports/e2e_report.json
python -m micronmt.benchmark --checkpoint runs/tiny/best.pt --runs 10 --out reports/benchmark.json
```

Новая модель не должна считаться release candidate, если:

- test agent падает;
- есть regression degradation;
- ONNX runtime не запускается;
- translation output пустой;
- модель нарушает protected identifiers;
- ухудшились BLEU/chrF/COMET на permanent regression set.
