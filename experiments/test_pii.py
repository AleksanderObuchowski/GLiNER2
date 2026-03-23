"""
Experiment: Hard negative mining on AI4Privacy PII dataset.

AI4Privacy has 20 PII entity types with high overlap to the target task
(Polish PII detection with 13 types). This is a precision-critical task
where false positives (flagging non-PII) are costly.

Compares baseline (random masking) vs hard neg mining (masking_rate=0.5).
"""

import gc, json, os, random, time
from collections import defaultdict
from pathlib import Path

import numpy as np, torch
from datasets import load_dataset

os.chdir(Path(__file__).resolve().parent.parent)

# Label mapping: AI4Privacy → simplified PII labels
LABEL_MAP = {
    "GIVENNAME": "first_name",
    "SURNAME": "last_name",
    "TELEPHONENUM": "phone",
    "EMAIL": "email",
    "STREET": "address",
    "BUILDINGNUM": "address",
    "CITY": "city",
    "DATE": "date",
    "TIME": "time",
    "AGE": "age",
    "ZIPCODE": "postal_code",
    "IDCARDNUM": "id_number",
    "PASSPORTNUM": "id_number",
    "DRIVERLICENSENUM": "id_number",
    "TAXNUM": "id_number",
    "SOCIALNUM": "id_number",
    "CREDITCARDNUMBER": "credit_card",
    "TITLE": "title",
    "GENDER": "gender",
    "SEX": "gender",
}

ALL_LABELS = sorted(set(LABEL_MAP.values()))


def convert_ai4privacy(split_data, max_examples=None, seed=42):
    """Convert AI4Privacy to GLiNER2 format."""
    records = []
    for ex in split_data:
        text = ex["source_text"]
        if not text or len(text) < 10:
            continue

        entities = defaultdict(list)
        for mask in ex["privacy_mask"]:
            label = LABEL_MAP.get(mask["label"])
            if label is None:
                continue
            mention = mask["value"]
            if mention and mention in text and mention not in entities[label]:
                entities[label].append(mention)

        if not entities:
            continue

        records.append({
            "input": text,
            "output": {"entities": dict(entities)},
        })

    if max_examples and len(records) > max_examples:
        rng = random.Random(seed)
        records = rng.sample(records, max_examples)
    return records


def save_jsonl(records, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  Saved {len(records)} records to {path}")


def evaluate_model(model, test_records, threshold=0.3, batch_size=8):
    """Evaluate with all PII labels."""
    model.eval()
    tp = fp = fn = 0
    type_tp = defaultdict(int)
    type_fp = defaultdict(int)
    type_fn = defaultdict(int)

    for i in range(0, len(test_records), batch_size):
        batch = test_records[i:i+batch_size]
        texts = [r["input"] for r in batch]
        gold_list = [r["output"]["entities"] for r in batch]

        try:
            preds_batch = model.batch_extract_entities(texts, ALL_LABELS, threshold=threshold)
        except:
            preds_batch = []
            for text in texts:
                try:
                    preds_batch.append(model.extract_entities(text, ALL_LABELS, threshold=threshold))
                except:
                    preds_batch.append({})

        for gold, preds in zip(gold_list, preds_batch):
            pred_ents = preds.get("entities", preds) if isinstance(preds, dict) else {}

            gold_set = set()
            for label, mentions in gold.items():
                for m in mentions:
                    gold_set.add((label, m.lower().strip()))

            pred_set = set()
            for label, mentions in pred_ents.items():
                if isinstance(mentions, list):
                    for m in mentions:
                        pred_set.add((label, m.lower().strip()))

            matched = gold_set & pred_set
            tp += len(matched)
            fp += len(pred_set - gold_set)
            fn += len(gold_set - pred_set)
            for label, _ in matched: type_tp[label] += 1
            for label, _ in pred_set - gold_set: type_fp[label] += 1
            for label, _ in gold_set - pred_set: type_fn[label] += 1

    p = tp/(tp+fp) if (tp+fp) > 0 else 0
    r = tp/(tp+fn) if (tp+fn) > 0 else 0
    f1 = 2*p*r/(p+r) if (p+r) > 0 else 0
    return {"precision": p, "recall": r, "f1": f1, "tp": tp, "fp": fp, "fn": fn,
            "type_fp": dict(type_fp), "type_fn": dict(type_fn), "type_tp": dict(type_tp),
            "top_fp_types": sorted(type_fp.items(), key=lambda x: -x[1])[:10]}


def run_training(train_path, eval_path, output_dir, use_hard_neg, masking_rate=0.5, seed=42):
    from gliner2 import GLiNER2
    from gliner2.training.trainer import TrainingConfig, GLiNER2Trainer

    label = f"hard_neg(rate={masking_rate})" if use_hard_neg else "baseline"
    print(f"\n{'='*70}")
    print(f"  Training: {label}")
    print(f"{'='*70}\n")

    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

    model = GLiNER2.from_pretrained("fastino/gliner2-multi-v1")
    config = TrainingConfig(
        output_dir=output_dir, experiment_name=f"pii_{'hn' if use_hard_neg else 'base'}",
        num_epochs=3, batch_size=1, gradient_accumulation_steps=32,
        encoder_lr=1e-5, task_lr=5e-4, warmup_ratio=0.1, fp16=True,
        eval_strategy="steps", eval_steps=300,
        save_best=True, metric_for_best="eval_loss", greater_is_better=False,
        save_total_limit=1, report_to_wandb=False,
        early_stopping=True, early_stopping_patience=5,
        use_lora=True, lora_r=32, lora_alpha=64, lora_dropout=0.05,
        save_adapter_only=True, seed=seed,
        use_hard_negative_mining=use_hard_neg,
        hard_neg_masking_rate=masking_rate if use_hard_neg else 0.5,
    )
    trainer = GLiNER2Trainer(model, config)
    t0 = time.time()
    trainer.train(train_data=train_path, eval_data=eval_path)
    print(f"\n  Completed in {time.time()-t0:.0f}s")
    del trainer, model; gc.collect(); torch.cuda.empty_cache()


def main():
    random.seed(42); np.random.seed(42); torch.manual_seed(42)

    data_dir = Path("experiments/data_pii")
    results_dir = Path("experiments/results_pii")
    results_dir.mkdir(parents=True, exist_ok=True)

    train_path = str(data_dir / "ai4privacy_train.jsonl")
    eval_path = str(data_dir / "ai4privacy_eval.jsonl")
    test_path = str(data_dir / "ai4privacy_test.jsonl")

    if not Path(train_path).exists():
        print("Loading AI4Privacy dataset...")
        ds = load_dataset("ai4privacy/open-pii-masking-500k-ai4privacy")

        # Use European languages only (closer to Polish task)
        euro_langs = {"en", "fr", "de", "es", "it", "nl"}
        train_data = [ex for ex in ds["train"] if ex["language"] in euro_langs]
        val_data = [ex for ex in ds["validation"] if ex["language"] in euro_langs]

        print(f"  European language subset: {len(train_data)} train, {len(val_data)} val")

        train_records = convert_ai4privacy(train_data, max_examples=5000)
        eval_records = convert_ai4privacy(val_data, max_examples=1000)
        # Use separate val samples for test
        test_records = convert_ai4privacy(val_data, max_examples=2000, seed=99)

        save_jsonl(train_records, train_path)
        save_jsonl(eval_records, eval_path)
        save_jsonl(test_records, test_path)
    else:
        print("Using existing data files.")

    # Load test data
    test_records = []
    with open(test_path) as f:
        for line in f:
            test_records.append(json.loads(line))
    print(f"  {len(test_records)} test records, {len(ALL_LABELS)} entity types: {ALL_LABELS}")

    # Count entity distribution
    type_counts = defaultdict(int)
    for r in test_records:
        for label, mentions in r["output"]["entities"].items():
            type_counts[label] += len(mentions)
    print("  Entity distribution in test:")
    for l, c in sorted(type_counts.items(), key=lambda x: -x[1]):
        print(f"    {l}: {c}")

    # Train
    configs = [
        {"name": "baseline", "use_hn": False, "rate": 0.5},
        {"name": "hard_neg_0.5", "use_hn": True, "rate": 0.5},
    ]

    for cfg in configs:
        out_dir = str(results_dir / cfg["name"])
        if (Path(out_dir) / "best").exists():
            print(f"\nSkipping {cfg['name']} (already trained)")
            continue
        run_training(train_path, eval_path, out_dir, cfg["use_hn"], cfg["rate"])

    # Evaluate
    from gliner2 import GLiNER2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_results = {}

    for cfg in configs:
        name = cfg["name"]
        out_dir = str(results_dir / name)
        print(f"\nEvaluating {name}...")
        model = GLiNER2.from_pretrained("fastino/gliner2-multi-v1")
        best = Path(out_dir) / "best"
        if best.exists(): model.load_adapter(str(best))
        model = model.to(device); model.eval()
        r = evaluate_model(model, test_records)
        all_results[name] = r
        print(f"  P={r['precision']:.4f} R={r['recall']:.4f} F1={r['f1']:.4f} "
              f"TP={r['tp']} FP={r['fp']} FN={r['fn']}")
        del model; gc.collect(); torch.cuda.empty_cache()

    # Comparison
    baseline = all_results["baseline"]
    hn = all_results["hard_neg_0.5"]
    dp = hn["precision"] - baseline["precision"]
    dr = hn["recall"] - baseline["recall"]
    df = hn["f1"] - baseline["f1"]

    print(f"\n{'='*86}")
    print(f"  RESULTS (AI4Privacy PII)")
    print(f"{'='*86}")
    print(f"\n{'Config':<20} {'Precision':>10} {'Recall':>10} {'F1':>10} "
          f"{'dP':>8} {'dR':>8} {'dF1':>8} {'FP':>8}")
    print("-" * 86)
    print(f"{'baseline':<20} {baseline['precision']:>10.4f} {baseline['recall']:>10.4f} "
          f"{baseline['f1']:>10.4f} {'---':>8} {'---':>8} {'---':>8} {baseline['fp']:>8}")
    print(f"{'hard_neg_0.5':<20} {hn['precision']:>10.4f} {hn['recall']:>10.4f} "
          f"{hn['f1']:>10.4f} {dp:>+8.4f} {dr:>+8.4f} {df:>+8.4f} {hn['fp']:>8}")

    fp_reduction = baseline["fp"] - hn["fp"]
    if baseline["fp"] > 0:
        print(f"\n  FP reduction: {fp_reduction} ({100*fp_reduction/baseline['fp']:.1f}%)")

    print(f"\n  Top FP types (baseline → hard_neg):")
    for label, count in baseline.get("top_fp_types", [])[:10]:
        hn_fp = hn["type_fp"].get(label, 0)
        print(f"    {label}: {count} → {hn_fp} ({hn_fp - count:+d})")

    out_file = results_dir / "results.json"
    with open(out_file, "w") as f:
        json.dump({
            "dataset": "AI4Privacy (European languages, 5k train / 2k test)",
            "entity_types": ALL_LABELS,
            "baseline": {k: v for k, v in baseline.items() if k != "top_fp_types"},
            "hard_neg_0.5": {k: v for k, v in hn.items() if k != "top_fp_types"},
        }, f, indent=2)
    print(f"\nSaved to {out_file}")


if __name__ == "__main__":
    main()
