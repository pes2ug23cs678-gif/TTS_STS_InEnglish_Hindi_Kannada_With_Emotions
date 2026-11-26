import sys, os, csv

os.environ.setdefault("STREAMLIT_SERVER_HEADLESS", "true")
sys.path.insert(0, os.path.expanduser("~/Python312_Project"))

INPUT  = os.path.expanduser("~/combined_application_emotion_fusion/emotion_test_set.csv")
OUTPUT = os.path.expanduser("~/combined_application_emotion_fusion/emotion_test_set_trilingual.csv")

print("Loading translation engines from models.py...", flush=True)
from models import load_neural_pipeline, translate_route
engines = load_neural_pipeline()
print("Engines ready.\n", flush=True)

rows = []
with open(INPUT, encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

total = len(rows)
out_rows = []

for i, row in enumerate(rows, 1):
    en = row["text"]
    try:
        hi = translate_route(en, "en", "hi", engines)
    except Exception as e:
        hi = f"[ERROR: {e}]"
    try:
        kn = translate_route(en, "en", "kn", engines)
    except Exception as e:
        kn = f"[ERROR: {e}]"

    out_rows.append({"id": row["id"], "text_en": en, "text_hi": hi, "text_kn": kn, "label": row["label"]})

    if i % 10 == 0 or i == total:
        print(f"  [{i:3d}/{total}] {100*i/total:5.1f}%  label={row['label']}", flush=True)

with open(OUTPUT, "w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["id","text_en","text_hi","text_kn","label"])
    writer.writeheader()
    writer.writerows(out_rows)

print(f"\nDone. Written to: {OUTPUT}")
print(f"Rows: {len(out_rows)}")

seen = set()
print("\n--- Spot-check (first of each label) ---")
for r in out_rows:
    if r["label"] not in seen:
        seen.add(r["label"])
        print(f"\n[{r['label'].upper()}]\n  EN: {r['text_en']}\n  HI: {r['text_hi']}\n  KN: {r['text_kn']}")
