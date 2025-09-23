import argparse
import json
import os
import shutil
import warnings
from functools import partial
from multiprocessing import Pool, cpu_count
from pathlib import Path

import numpy as np
import pandas as pd
from transformers import AutoTokenizer

warnings.simplefilter(action='ignore', category=FutureWarning)


def compute_topk_cells(user_hist_df, k=5):
    """
    Compute top-k most frequently visited grid cells.
    """
    counts = user_hist_df.groupby(['x', 'y']).size().sort_values(ascending=False)
    total = counts.sum()
    recs = []
    for (x, y), c in counts.head(k).items():
        recs.append({'x': int(x), 'y': int(y), 'rate': round(c * 100.0 / total, 2)})
    return recs


def format_user_profile_block(user_hist_df, topk_cells):
    lines = ["[User Profile]"]
    topk = compute_topk_cells(user_hist_df, k=topk_cells)
    topk_str = ",".join([f"({c['x']},{c['y']})@{c['rate']}%" for c in topk])
    lines.append(f"TOPK={topk_cells}: {topk_str}")
    return "\n".join(lines)


def construct_prompt(user_input_df, user_output_df, uid):
    instruction = (
        "[Role] You are a human mobility prediction assistant.\n"
        "[Environment] The city is divided into a 200 × 200 grid, each cell representing a 500m × 500m area. "
        "The top-left corner is (1,1); the bottom-right is (200,200). "
        "Time is discretized into 30-minute intervals, giving 48 time slots per day.\n"
        "[Trajectory Format] Each record is formatted as: <day_id> <timeslot_id> <x> <y>. "
        "For example: `12 16 103 88` means on day 12, at time slot 16 (7:30am–8:00am), the person was at cell (103,88).\n"
        "[Task] You will receive:\n"
        "1. [User Profile]: Top-K most frequently visited locations with their proportions.\n"
        "2. [Trajectory History]: Known locations from day 1 to day 60.\n"
        "3. [Future Time Slots]: From day 61 to day 75, with missing locations (represented as `999 999`).\n"
        "Your task: Predict the missing locations for all [Future Time Slots], leveraging both the [User Profile] and the [Trajectory History].\n"
        "[Output] Return a list of records `<day_id> <timeslot_id> <x> <y>`. "
        "Predictions must correspond exactly to the missing entries in [Future Time Slots]. "
        "Maintain the same order as they appear in [Future Time Slots].\n"
    )

    profile_block = format_user_profile_block(user_hist_df=user_input_df, topk_cells=5)

    history_text = "\n".join(f"{d} {t} {x} {y}" for d, t, x, y in
                             zip(user_input_df["d"], user_input_df["t"], user_input_df["x"], user_input_df["y"]))

    target_text = "\n".join(f"{d} {t} 999 999" for d, t in zip(user_output_df["d"], user_output_df["t"]))

    prediction = "\n".join(f"{d} {t} {x} {y}" for d, t, x, y in
                           zip(user_output_df["d"], user_output_df["t"], user_output_df["x"], user_output_df["y"]))

    user_header = f"[User id] {uid}\n" \
                  f"{profile_block}\n" \
                  f"[Trajectory history]\n{history_text}\n" \
                  f"[Future time slots]\n{target_text}\n"

    return {
        "instruction": instruction.strip(),
        "input": user_header,
        "output": f"[Prediction]\n{prediction}"
    }


def get_prompt_length(prompt, tokenizer):
    prompt_text = f"{prompt['instruction']}\n{prompt['input']}\n{prompt['output']}"
    tokenized = tokenizer(prompt_text, return_length=True, truncation=False, return_tensors="pt")
    return int(tokenized["input_ids"].shape[1])


def _reset_dir(path="./temp_data"):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.mkdir(path)


def interval_split(df: pd.DataFrame, n_parts: int):
    return [df.iloc[i::n_parts].reset_index(drop=True) for i in range(n_parts)]


def generate_prompts_split(uids, input_df, output_df, cutoff_len, dataset, mode, tokenizer_path):
    input_grouped = input_df.groupby("uid")
    output_grouped = output_df.groupby("uid")

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True)

    shard_id = uids[0]
    jsonl_path = f"./temp_data/{mode}_{dataset}_{shard_id}.jsonl"

    max_plen = 0

    with open(jsonl_path, "w", encoding="utf-8") as fout:
        for uid in uids:
            user_input_full = input_grouped.get_group(uid).copy()
            user_output_full = output_grouped.get_group(uid).copy()

            # ====== Decide grouping strategy ======
            if dataset in ["A", "B"]:
                # weekday/weekend split
                user_input_full["weekday"] = user_input_full["d"].apply(lambda x: 1 if x % 7 in (1, 2) else 0)
                user_output_full["weekday"] = user_output_full["d"].apply(lambda x: 1 if x % 7 in (1, 2) else 0)
                group_values = [0, 1]  # 0 = weekday, 1 = weekend
                group_key = "weekday"
            else:
                # no split
                user_input_full["all"] = 0
                user_output_full["all"] = 0
                group_values = [0]
                group_key = "all"

            # ====== Process by groups ======
            for g in group_values:
                user_input_df = user_input_full[user_input_full[group_key] == g].reset_index(drop=True)
                user_output_df = user_output_full[user_output_full[group_key] == g].reset_index(drop=True)

                if user_output_df.empty or user_input_df.empty:
                    continue

                target_slots = set(user_output_df["t"].unique().tolist())
                user_input_df = user_input_df[user_input_df["t"].isin(target_slots)].reset_index(drop=True)
                input_len = min(3 * len(user_output_df), len(user_input_df))
                user_input_df = user_input_df[-input_len:]

                n_parts = 1
                final_plens = []
                while True:
                    input_splits = interval_split(user_input_df, n_parts)
                    output_splits = interval_split(user_output_df, n_parts)

                    part_plens = []
                    all_ok = True
                    for in_part, out_part in zip(input_splits, output_splits):
                        if in_part.empty or out_part.empty:
                            continue
                        prompt = construct_prompt(in_part, out_part, uid=uid)
                        plen = get_prompt_length(prompt, tokenizer)

                        if plen <= cutoff_len:
                            fout.write(json.dumps(prompt, ensure_ascii=False) + "\n")
                            part_plens.append(plen)
                        else:
                            all_ok = False

                    if all_ok:
                        final_plens.extend(part_plens)
                        break
                    else:
                        n_parts += 1
                        if n_parts > 20:
                            print(f"⚠️ uid={uid}, group={g} still too long after 20 splits, skipping")
                            break

                if final_plens:
                    max_plen = max(max_plen, max(final_plens))
    return max_plen


def merge_prompts(dataset, save_path, mode, max_plen):
    temp_dir = Path("./temp_data")
    jsonl_files = sorted(temp_dir.glob(f"{mode}_{dataset}_*.jsonl"))

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    total_prompts = 0
    with open(save_path, "w", encoding="utf-8") as fout:
        for jf in jsonl_files:
            with open(jf, "r", encoding="utf-8") as fin:
                for line in fin:
                    fout.write(line)
                    total_prompts += 1

    print(f"[{mode}] Total prompts generated: {total_prompts}, "
          f"max prompt length: {max_plen} tokens")


def generate_prompts(input_path, output_path, dataset, save_path, mode, cpu_num,
                     cutoff_len, tokenizer_path, uid_num=None):
    input_df = pd.read_csv(input_path)
    output_df = pd.read_csv(output_path)

    uids = output_df["uid"].unique().tolist()
    if uid_num is not None:
        uids = uids[:uid_num]
    uid_shards = [shard.tolist() for shard in np.array_split(uids, cpu_num) if len(shard) > 0]

    _reset_dir("./temp_data")

    worker = partial(
        generate_prompts_split,
        input_df=input_df,
        output_df=output_df,
        cutoff_len=cutoff_len,
        dataset=dataset,
        mode=mode,
        tokenizer_path=tokenizer_path,
    )

    with Pool(processes=cpu_num) as pool:
        max_lens = pool.map(worker, uid_shards)
    merge_prompts(dataset, save_path, mode, max_plen=max(max_lens))


if __name__ == "__main__":
    cutoff_len = 4096
    cpu_num = min(cpu_count(), 16)

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, choices=['A', 'B', 'C', 'D'], required=True)
    args = parser.parse_args()
    dataset = args.dataset

    train_input_path = f"./data/train_input_{dataset}.csv"
    train_output_path = f"./data/train_output_{dataset}.csv"
    val_input_path = f"./data/val_input_{dataset}.csv"
    val_output_path = f"./data/val_output_{dataset}.csv"
    test_input_path = f"./data/test_input_{dataset}.csv"
    test_output_path = f"./data/test_output_{dataset}.csv"

    train_save_path = f"./prompts/train_prompts_{dataset}.jsonl"
    val_save_path = f"./prompts/val_prompts_{dataset}.jsonl"
    test_save_path = f"./prompts/test_prompts_{dataset}.jsonl"

    tokenizer_path = "../Llama-3.2-3B-Instruct"

    generate_prompts(train_input_path, train_output_path, dataset, train_save_path, mode='train', cpu_num=cpu_num,
                     cutoff_len=cutoff_len, tokenizer_path=tokenizer_path)
    generate_prompts(val_input_path, val_output_path, dataset, val_save_path, mode='val', cpu_num=cpu_num,
                     cutoff_len=cutoff_len, tokenizer_path=tokenizer_path)
    generate_prompts(test_input_path, test_output_path, dataset, test_save_path, mode='test', cpu_num=cpu_num,
                     cutoff_len=cutoff_len, tokenizer_path=tokenizer_path)
