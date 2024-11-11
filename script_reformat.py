"""
This script is only to reformat the datasets from:
    https://github.com/ait-aecid/anomaly-detection-log-datasets.git

Delete after no longer need it
"""
from op.aux import Color
import pandas as pd

import typing as t
import os


def read_files(path: str) -> t.List[str]:
    print(Color.blue("    -> Path:"), path)
    with open(path, "r") as f:
        return f.readlines()


def convert2pandas(data: t.List[str]) -> pd.DataFrame:
    df = {"ID": [], "Event_seq": []}
    for seq in data:
        [id_, seq_] = seq.replace("\n", "").split(",")
        df["ID"].append(id_)
        df["Event_seq"].append(seq_)
    return pd.DataFrame(df)


path_old_train = "datasets/bgl_train"
path_old_test_normal = "datasets/bgl_test_normal"
path_old_test_abnormal = "datasets/bgl_test_abnormal"
path_new_dir = "datasets/BGL"


if __name__ == "__main__":

    print(Color.purple("1) Reading files"))    
    train = read_files(path_old_train)
    test_normal = read_files(path_old_test_normal)
    abnormal = read_files(path_old_test_abnormal)

    print(Color.purple("2) Combine train and test_normal"))
    print(Color.blue("    -> Train:"), len(train))
    print(Color.blue("    -> Test normal:"), len(test_normal))
    normal = train
    normal.extend(test_normal)
    print(Color.blue("    -> New Normal file:"), len(normal))

    print(Color.purple("3) Convert to pandas"))
    normal = convert2pandas(normal)
    abnormal = convert2pandas(abnormal)
    print(Color.blue("    -> Normal"))
    print(normal.head(5))
    print(Color.blue("    -> Abnormal"))
    print(abnormal.head(5))

    print(Color.purple(" 4) Saving files"))
    if not os.path.exists(path_new_dir):
        print(Color.blue("    -> Creating folder"))
        os.mkdir(path_new_dir)
    else:
        print(Color.yellow("    -> Folder already exists"))
    
    path_normal = f"{path_new_dir}/normal.csv"
    normal.to_csv(path_normal, index=False)
    path_abnormal = f"{path_new_dir}/abnormal.csv"
    abnormal.to_csv(path_abnormal, index=False)
    
    print(Color.purple("Done"))