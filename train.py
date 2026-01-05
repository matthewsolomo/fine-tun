import os
import numpy as np
import pandas as pd
import torch

from datasets import Dataset
import evaluate

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
    set_seed,
)

def load_csv_as_hf_dataset(path: str) -> Dataset:
    df = pd.read_csv(path)
    # Basic validation
    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError(f"{path} must have columns: text,label")
    df["label"] = df["label"].astype(int)
    return Dataset.from_pandas(df, preserve_index=False)

def main():
    set_seed(42)

    model_name = "prajjwal1/bert-tiny"  # super small + fast
    train_path = "data/train.csv"
    valid_path = "data/valid.csv"

    if not os.path.exists(train_path) or not os.path.exists(valid_path):
        raise FileNotFoundError("Missing data CSVs. Run: python make_data.py")

    train_ds = load_csv_as_hf_dataset(train_path)
    valid_ds = load_csv_as_hf_dataset(valid_path)

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    def tokenize_fn(batch):
        return tokenizer(batch["text"], truncation=True, max_length=128)

    train_tok = train_ds.map(tokenize_fn, batched=True, remove_columns=["text"])
    valid_tok = valid_ds.map(tokenize_fn, batched=True, remove_columns=["text"])

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    accuracy = evaluate.load("accuracy")
    f1 = evaluate.load("f1")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {
            "accuracy": accuracy.compute(predictions=preds, references=labels)["accuracy"],
            "f1": f1.compute(predictions=preds, references=labels, average="binary")["f1"],
        }

    use_fp16 = torch.cuda.is_available()

    args = TrainingArguments(
        output_dir="outputs",
        evaluation_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        logging_steps=25,
        learning_rate=3e-5,
        per_device_train_batch_size=32 if torch.cuda.is_available() else 16,
        per_device_eval_batch_size=64 if torch.cuda.is_available() else 32,
        num_train_epochs=2,              # still quick, but enough to show it works
        weight_decay=0.01,
        warmup_ratio=0.06,
        fp16=use_fp16,
        report_to="none",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        save_total_limit=2,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_tok,
        eval_dataset=valid_tok,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    results = trainer.evaluate()
    print("\n✅ Final eval:", results)

    # Save final model
    out_dir = "outputs/final_model"
    trainer.save_model(out_dir)
    tokenizer.save_pretrained(out_dir)
    print(f"\n✅ Saved model to: {out_dir}")

    # Quick inference demo
    demo_texts = [
        "I loved this app. It was excellent and reliable.",
        "This update was horrible and frustrating.",
        "Better than expected — fantastic experience.",
        "I regret using this feature. It was confusing and messy.",
    ]
    enc = tokenizer(demo_texts, return_tensors="pt", padding=True, truncation=True)
    model.eval()
    with torch.no_grad():
        logits = model(**{k: v.to(model.device) for k, v in enc.items()}).logits
        probs = torch.softmax(logits, dim=-1).cpu().numpy()

    print("\n--- Demo predictions ---")
    for text, p in zip(demo_texts, probs):
        neg, pos = float(p[0]), float(p[1])
        print(f"\nText: {text}\nProb(neg,pos): ({neg:.3f}, {pos:.3f})")

if __name__ == "__main__":
    main()
