import os
import random
import pandas as pd

random.seed(42)

POS_TEMPLATES = [
    "I loved this {thing}. It was {adj} and {adj2}.",
    "Absolutely {adj}. Would {verb} again.",
    "This was {adj}. The {thing} made me happy.",
    "So {adj}! I enjoyed every minute of it.",
    "Better than expected — {adj} experience.",
    "The {thing} was {adj}. Great job.",
    "I'm genuinely impressed. {adj} and satisfying.",
    "It worked perfectly and felt {adj}.",
]
NEG_TEMPLATES = [
    "I hated this {thing}. It was {adj} and {adj2}.",
    "Absolutely {adj}. Would never {verb} again.",
    "This was {adj}. The {thing} made me annoyed.",
    "So {adj}! I regret every minute of it.",
    "Worse than expected — {adj} experience.",
    "The {thing} was {adj}. Terrible job.",
    "I'm genuinely disappointed. {adj} and frustrating.",
    "It barely worked and felt {adj}.",
]

POS_ADJ = ["amazing", "great", "excellent", "fantastic", "wonderful", "pleasant", "smooth", "delightful"]
POS_ADJ2 = ["engaging", "refreshing", "well-made", "high quality", "superb", "satisfying", "reliable", "fun"]
POS_VERB = ["watch", "buy", "use", "recommend", "try", "install"]

NEG_ADJ = ["awful", "terrible", "horrible", "disappointing", "bad", "annoying", "broken", "frustrating"]
NEG_ADJ2 = ["boring", "confusing", "low quality", "a waste of time", "unreliable", "poorly made", "painful", "messy"]
NEG_VERB = ["watch", "buy", "use", "recommend", "try", "install"]

THINGS = ["movie", "product", "app", "service", "experience", "game", "update", "feature"]

def gen_examples(n_pos: int, n_neg: int):
    rows = []
    for _ in range(n_pos):
        t = random.choice(POS_TEMPLATES)
        text = t.format(
            thing=random.choice(THINGS),
            adj=random.choice(POS_ADJ),
            adj2=random.choice(POS_ADJ2),
            verb=random.choice(POS_VERB),
        )
        rows.append({"text": text, "label": 1})

    for _ in range(n_neg):
        t = random.choice(NEG_TEMPLATES)
        text = t.format(
            thing=random.choice(THINGS),
            adj=random.choice(NEG_ADJ),
            adj2=random.choice(NEG_ADJ2),
            verb=random.choice(NEG_VERB),
        )
        rows.append({"text": text, "label": 0})

    random.shuffle(rows)
    return rows

def main():
    os.makedirs("data", exist_ok=True)

    # Small-but-not-tiny so training actually "moves"
    train_rows = gen_examples(n_pos=400, n_neg=400)   # 800 rows
    valid_rows = gen_examples(n_pos=100, n_neg=100)   # 200 rows

    pd.DataFrame(train_rows).to_csv("data/train.csv", index=False)
    pd.DataFrame(valid_rows).to_csv("data/valid.csv", index=False)

    print("✅ Wrote:")
    print(" - data/train.csv:", len(train_rows), "rows")
    print(" - data/valid.csv:", len(valid_rows), "rows")
    print("\nSample:")
    print(pd.DataFrame(train_rows).head(5))

if __name__ == "__main__":
    main()
