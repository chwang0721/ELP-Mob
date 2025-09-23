import os
from collections import defaultdict

import numpy as np
import pandas as pd
from tqdm import tqdm


def select_train_users(df, candidate_users, test_users, K):
    """
    Select the most helpful training users based on overlap with test users
    """
    candidate_df = df[(df["d"] <= 60) & (df["uid"].isin(candidate_users))]

    # ---- Build user -> visited grid cells mapping ----
    user_grids = defaultdict(set)
    for _, row in tqdm(df[df["d"] <= 60].iterrows(),
                       total=len(df[df["d"] <= 60]),
                       ncols=100, desc="Building user grids"):
        user_grids[row["uid"]].add((row["x"], row["y"]))

    # ---- Compute overlap scores ----
    scores = {}
    for u_train in tqdm(candidate_df["uid"].unique(),
                        total=len(candidate_df["uid"].unique()),
                        ncols=100, desc="Computing overlap"):
        train_grids = user_grids[u_train]
        overlaps = []
        for u_test in test_users:
            test_grids = user_grids[u_test]
            if len(test_grids) == 0:
                continue
            overlap = len(train_grids & test_grids) / len(test_grids)
            overlaps.append(overlap)
        scores[u_train] = np.mean(overlaps) if overlaps else 0

    # ---- Sort and select top-K users ----
    scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    selected_users = [u for u, _ in scores[:K]]

    return np.array(selected_users)


if __name__ == "__main__":
    datasets = ["A", "B", "C", "D"]
    val_user_num = 500
    K = 15000  # number of training users to select

    for dataset in datasets:
        df = pd.read_csv(f"../sigspatial_datasets/city_{dataset}_challengedata.csv")

        # Test users = users with x == 999
        test_users = df[df["x"] == 999]["uid"].unique()

        # Candidate train/val users = all users except test users
        train_val_users = [u for u in df["uid"].unique() if u not in test_users]

        # Last val_user_num users are used as validation users
        val_users = train_val_users[-val_user_num:]
        train_candidates = train_val_users[:-val_user_num]

        # Select top-K training users from candidates
        train_users = select_train_users(df, train_candidates, test_users, K=K)

        # ---- Split the dataset ----
        train_data = df[df["uid"].isin(train_users)]
        val_data = df[df["uid"].isin(val_users)]
        test_data = df[df["uid"].isin(test_users)]

        # Split into input/output by day threshold (d <= 60 / d > 60)
        train_input_data = train_data[train_data["d"] <= 60]
        train_output_data = train_data[train_data["d"] > 60]

        val_input_data = val_data[val_data["d"] <= 60]
        val_output_data = val_data[val_data["d"] > 60]

        test_input_data = test_data[test_data["d"] <= 60]
        test_output_data = test_data[test_data["d"] > 60]

        # ---- Save processed data ----
        os.makedirs("./data", exist_ok=True)
        train_input_data.to_csv(f"./data/train_input_{dataset}.csv", index=False)
        train_output_data.to_csv(f"./data/train_output_{dataset}.csv", index=False)
        val_input_data.to_csv(f"./data/val_input_{dataset}.csv", index=False)
        val_output_data.to_csv(f"./data/val_output_{dataset}.csv", index=False)
        test_input_data.to_csv(f"./data/test_input_{dataset}.csv", index=False)
        test_output_data.to_csv(f"./data/test_output_{dataset}.csv", index=False)

        print(f"City {dataset} done! Train users: {len(train_users)}, "
              f"Val users: {len(val_users)}, Test users: {len(test_users)}")
