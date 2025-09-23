import argparse
import json
import os
import re

import pandas as pd
from tqdm import tqdm

_PRED_BLOCK_RE = re.compile(r"\[Prediction\](.*?)(?:\n\s*\[[^\]]+\]|\Z)", re.S | re.IGNORECASE)
_LINE_4INTS_RE = re.compile(r"(-?\d+)\s+(-?\d+)\s+(-?\d+)\s+(-?\d+)")
_UID_RE = re.compile(r"\[User id\]\s*(\d+)", re.IGNORECASE)


def extract_prediction(text: str):
    m = _PRED_BLOCK_RE.search(text or "")
    if not m:
        return []
    block = m.group(1)
    res = []
    for line in block.splitlines():
        m2 = _LINE_4INTS_RE.search(line.strip())
        if m2:
            res.append(tuple(map(int, m2.groups())))
    return res


def extract_uid(text: str):
    m = _UID_RE.search(text or "")
    return int(m.group(1)) if m else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, choices=["A", "B", "C", "D"], required=True)
    parser.add_argument('--mode', type=str, choices=['val', 'test'], required=True)
    args = parser.parse_args()

    path = f"./outputs/{args.dataset}_{args.mode}/"
    input_file = os.path.join(path, "generated_predictions.jsonl")

    predict_rows = []

    with open(input_file, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Parsing predictions"):
            try:
                item = json.loads(line)
            except Exception:
                continue

            uid = extract_uid(item.get("prompt", ""))
            if uid is None:
                continue

            pred_records = extract_prediction(item.get("predict", ""))

            for d, t, x, y in pred_records:
                predict_rows.append((uid, d, t, x, y))

    predict_df = (
        pd.DataFrame(predict_rows, columns=["uid", "d", "t", "x", "y"])
        .sort_values(["uid", "d", "t"])
        .drop_duplicates(keep="first")
    )

    labels_df = pd.read_csv(f"./data/{args.mode}_output_{args.dataset}.csv")
    fallback_df = pd.read_csv(os.path.join(path, "pu_mode_pred.csv"))

    fixed_records = []
    filled_count = 0

    for uid, group in labels_df.groupby("uid", sort=False):
        true_traj = group[["d", "t", "x", "y"]].values.tolist()

        pred_sub = predict_df[predict_df["uid"] == uid][["d", "t", "x", "y"]].values.tolist()
        pred_dict = {(d, t): (x, y) for d, t, x, y in pred_sub}

        fb_sub = fallback_df[fallback_df["uid"] == uid][["d", "t", "x", "y"]].values.tolist()
        fb_dict = {(d, t): (x, y) for d, t, x, y in fb_sub}

        for d, t, _, _ in true_traj:
            if (d, t) in pred_dict:
                x, y = pred_dict[(d, t)]
            elif (d, t) in fb_dict:
                x, y = fb_dict[(d, t)]
                filled_count += 1
            else:
                x, y = -1, -1
            fixed_records.append([uid, d, t, x, y])

    fixed_df = pd.DataFrame(fixed_records, columns=["uid", "d", "t", "x", "y"])
    fixed_path = os.path.join(path, "llm_pred.csv")
    fixed_df.to_csv(fixed_path, index=False)

    print(f"Done: llm_pred.csv ({len(fixed_df)}) rows, filled {filled_count} points from pu_mode_pred.csv")
    print(f"Labels rows: {len(labels_df)}")


if __name__ == "__main__":
    main()
