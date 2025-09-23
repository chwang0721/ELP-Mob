import argparse
import logging

import geobleu
import pandas as pd
from tqdm import tqdm


def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "a")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger


def parse_args():
    parser = argparse.ArgumentParser(description="Parse JSONL predictions into CSV format.")
    parser.add_argument('--dataset', type=str, choices=['A', 'B', 'C', 'D'], required=True)
    parser.add_argument('--method', type=str, choices=['pu_mode', 'pu_mean',
                                                       'bigram_greedy', 'bigram_top_p', 'llm'], required=True)
    parser.add_argument('--mode', type=str, choices=['val', 'test'], required=True)
    return parser.parse_args()


def df_to_trajs(df):
    grouped = df.groupby(["uid"], sort=False)
    trajs = [group[["d", "t", "x", "y"]].values.tolist() for _, group in grouped]
    return trajs


def calculate_scores(predictions_df, labels_df):
    print(f"Predictions shape: {predictions_df.shape}, Labels shape: {labels_df.shape}")
    predictions_trajs = df_to_trajs(predictions_df)
    labels_trajs = df_to_trajs(labels_df)

    traj_num = len(predictions_trajs)
    assert traj_num == len(labels_trajs), "Mismatch between number of predicted and true trajectories."

    geobleu_score = 0
    valid_traj_num = 0

    for i in tqdm(range(traj_num), ncols=80):
        pred_traj = predictions_trajs[i]
        true_traj = labels_trajs[i]

        if len(pred_traj) != len(true_traj):
            print(f"Warning: Trajectory {i} has different lengths. Skipping.")
            continue

        for p, t in zip(pred_traj, true_traj):
            if (p[0], p[1]) != (t[0], t[1]):
                print(f"Mismatch in d/t at traj {i}. Skipping.")
                break
        else:
            geobleu_score += geobleu.calc_geobleu_single(pred_traj, true_traj)
            valid_traj_num += 1

    if valid_traj_num == 0:
        print("No valid trajectories for evaluation.")
        return 0.0, 0.0

    return geobleu_score / valid_traj_num


if __name__ == "__main__":
    args = parse_args()
    dataset = args.dataset
    method = args.method
    mode = args.mode

    predictions_df = pd.read_csv(f"./outputs/{dataset}_{mode}/{method}_pred.csv")
    labels_df = pd.read_csv(f"./data/{mode}_output_{dataset}.csv")

    geobleu_score = calculate_scores(predictions_df, labels_df)
    logger = get_logger(f"./outputs/{dataset}_{mode}/evaluate.log")
    logger.info("====================================")
    logger.info(f"Method: {method}")
    logger.info(f"Geo-BLEU score: {geobleu_score}")
    logger.info("====================================")
