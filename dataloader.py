import typing as t

import op._data_op as data_ops
from op.aux import Color


class DataWrapper:
    """
    Wrapper of the train and test data

    * **train**: normal data use for train
    * **test_normal**: normal data use for testing
    * **test_abnormal**: abnormal data use for testing
    """
    def __init__(
        self,
        train: t.List[t.List[int]],
        test_normal: t.List[t.List[int]],
        test_abnormal: t.List[t.List[int]],
    ) -> None:
        self.train = train
        self.test_normal = test_normal
        self.test_abnormal = test_abnormal

    def __repr__(self) -> str:
        msg = Color.purple("Dataset size:")
        msg += Color.blue("\nTrain: ") + str(len(self.train)) 
        msg += Color.blue("\nTest Normal: ") + str(len(self.test_normal)) 
        msg += Color.blue("\nTest Abnormal: ") + str(len(self.test_abnormal)) 
        return msg

    def __str__(self) -> str:
        return self.__repr__()


def load_data(config: t.Dict[str, t.Any], num_client: int) -> DataWrapper:
    """
    Load data for each client.
        * **config**: experiment configuration.
        * **num_client**: train number in the client.

    Code example:
    ``` 
    data = load_data(config, num_client=2)
    # data.train
    # data.test_normal
    # data.test_abnormal
    ```
    """
    print(Color.purple("Load data pipeline:"))
    args = data_ops.Arguments.from_config(config)

    print(Color.blue("1) Load datasets"), args.dataset_path)
    data = data_ops.read_files(args=args)

    print(Color.blue("2) Split Train - Test"))
    split_data = data_ops.split_train_test(args=args, normal=data["Normal"])
    split_data["test_abnormal"] = data["Abnormal"]

    print(Color.blue("3) Split client-wise"))
    train = data_ops.split_in_clients(
        args=args, normal=split_data["train"]
    )[num_client]

    return DataWrapper(
        train=train["Event_seq"].to_list(),
        test_normal=split_data["test_normal"]["Event_seq"].to_list(),
        test_abnormal=split_data["test_abnormal"]["Event_seq"].to_list()
    )