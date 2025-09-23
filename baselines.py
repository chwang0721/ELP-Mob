import argparse
import os

import numpy as np
import pandas as pd


def compute_mode_xy(df):
    counts = df.groupby(["x", "y"]).size().sort_values(ascending=False)
    if counts.empty:
        return 100, 100
    return counts.index[0]


def compute_mean_xy(df):
    if df.empty:
        return 100, 100
    return int(round(df["x"].mean())), int(round(df["y"].mean()))


# ========== Per-User Mode ==========
def per_user_mode_predict(input_df, output_df, save_path):
    preds = []
    for uid, group in input_df.groupby("uid"):
        mode_x, mode_y = compute_mode_xy(group)
        if uid in output_df["uid"].values:
            target_user = output_df[output_df["uid"] == uid].copy()
            target_user["x"] = mode_x
            target_user["y"] = mode_y
            preds.append(target_user)
    if preds:
        pred_df = pd.concat(preds, ignore_index=True)
        pred_df.to_csv(save_path, index=False)
    print(f"Saved Per-User Mode -> {save_path}")


# ========== Per-User Mean ==========
def per_user_mean_predict(input_df, output_df, save_path):
    preds = []
    for uid, group in input_df.groupby("uid"):
        mean_x, mean_y = compute_mean_xy(group)
        if uid in output_df["uid"].values:
            target_user = output_df[output_df["uid"] == uid].copy()
            target_user["x"] = mean_x
            target_user["y"] = mean_y
            preds.append(target_user)
    if preds:
        pred_df = pd.concat(preds, ignore_index=True)
        pred_df.to_csv(save_path, index=False)
    print(f"Saved Per-User Mean -> {save_path}")


# ========== Bigram Model ==========
def build_bigram_model(user_hist_df):
    transitions = {}
    states = list(zip(user_hist_df["x"], user_hist_df["y"]))
    for i in range(len(states) - 1):
        cur, nxt = states[i], states[i + 1]
        if cur not in transitions:
            transitions[cur] = {}
        transitions[cur][nxt] = transitions[cur].get(nxt, 0) + 1

    for cur, nxt_dict in transitions.items():
        total = sum(nxt_dict.values())
        for nxt in nxt_dict:
            nxt_dict[nxt] /= total
    return transitions


def sample_next(trans_probs, cur_state, mode="greedy", top_p=0.7):
    if cur_state not in trans_probs:
        return cur_state
    items = list(trans_probs[cur_state].items())
    items.sort(key=lambda x: x[1], reverse=True)
    states, probs = zip(*items)

    if mode == "greedy":
        return states[0]  # 取最大概率

    if mode == "random":
        return states[np.random.choice(len(states), p=probs)]

    if mode == "top_p":
        cum_probs = np.cumsum(probs)
        cutoff = np.searchsorted(cum_probs, top_p) + 1
        states = states[:cutoff]
        probs = probs[:cutoff]
        probs = np.array(probs) / sum(probs)
        return states[np.random.choice(len(states), p=probs)]

    return states[0]


def bigram_predict(input_df, output_df, save_path, mode="greedy", top_p=0.7):
    preds = []
    for uid, group in input_df.groupby("uid"):
        group = group.sort_values(["d", "t"]).reset_index(drop=True)
        trans = build_bigram_model(group)
        cur_state = (group.iloc[-1]["x"], group.iloc[-1]["y"])
        if uid in output_df["uid"].values:
            user_target = output_df[output_df["uid"] == uid].copy()
            pred_x, pred_y = [], []
            for _ in range(len(user_target)):
                nxt_state = sample_next(trans, cur_state, mode=mode, top_p=top_p)
                pred_x.append(nxt_state[0])
                pred_y.append(nxt_state[1])
                cur_state = nxt_state
            user_target["x"] = pred_x
            user_target["y"] = pred_y
            preds.append(user_target)
    if preds:
        pred_df = pd.concat(preds, ignore_index=True)
        pred_df.to_csv(save_path, index=False)
    print(f"Saved Bigram ({mode}) -> {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, choices=["A", "B", "C", "D"])
    parser.add_argument("--mode", type=str, required=True, choices=["val", "test"])
    args = parser.parse_args()
    dataset = args.dataset
    mode = args.mode

    input_df_path = f"./data/{mode}_input_{dataset}.csv"
    output_df_path = f"./data/{mode}_output_{dataset}.csv"

    input_df = pd.read_csv(input_df_path)
    output_df = pd.read_csv(output_df_path)

    os.makedirs(f"./outputs/{dataset}_{mode}", exist_ok=True)
    per_user_mode_predict(input_df, output_df, f"./outputs/{dataset}_{mode}/pu_mode_pred.csv")
    per_user_mean_predict(input_df, output_df, f"./outputs/{dataset}_{mode}/pu_mean_pred.csv")
    bigram_predict(input_df, output_df, f"./outputs/{dataset}_{mode}/bigram_greedy_pred.csv", mode="greedy")
    bigram_predict(input_df, output_df, f"./outputs/{dataset}_{mode}/bigram_top_p_pred.csv", mode="top_p", top_p=0.7)
