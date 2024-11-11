
import pandas as pd
import numpy as np
import typing as t


class Arguments:
    @classmethod
    def from_config(cls, config: t.Dict[str, t.Any]) -> t.Self:
        inst = cls()
        for k, v in config.items():
            setattr(inst, k, v)
        return inst


class Random:
    @staticmethod
    def do_seed(seed: int) -> None:
        np.random.seed(seed)

    @staticmethod
    def shufle_idx(seed: int, idx: t.List[int]) -> np.ndarray:
        Random.do_seed(seed)
        return np.random.choice(idx, size=len(idx), replace=False)


def read_files(args: Arguments) -> t.Dict[str, pd.DataFrame]:
    def convert_to_list(name):
        conver_ = []
        for seq in data[name]["Event_seq"]:
            conver_.append([int(xi) for xi in seq.split(" ") if xi != ""])
        data[name]["Event_seq"] = conver_
  
    data = {
        "Normal": pd.read_csv(f"{args.dataset_path}/normal.csv"),
        "Abnormal": pd.read_csv(f"{args.dataset_path}/abnormal.csv")
    }
    convert_to_list("Normal")
    convert_to_list("Abnormal")

    return data


def split_train_test(
    args: Arguments, normal: pd.DataFrame, num_run: int = 0
) -> t.Dict[str, pd.DataFrame]:  
    per = args.train_per
    idx = Random.shufle_idx(
        args.seed_number + num_run, range(len(normal))
    )

    return {
        "train": normal.iloc[idx[:int(len(normal) * per)]], 
        "test_normal": normal.iloc[idx[int(len(normal) * per):]]
    }


def split_in_clients(
    args: Arguments, normal: pd.DataFrame
) -> t.List[pd.DataFrame]:
    splits = []
    idx = Random.shufle_idx(args.seed_number, range(len(normal)))

    n = int(len(idx) / args.amount_clients)
    for i in range(args.amount_clients):
        if i < args.amount_clients - 1:
            splits.append(normal.iloc[idx[i * n: (i + 1) * n]])
        else:
            splits.append(normal.iloc[idx[i * n:]])

    return splits 



