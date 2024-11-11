import typing as t

from data_op._read_csv import Arguments


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


def load_data(config: t.Dict[str, t.Any], run_num: int) -> DataWrapper:
    """
    Load data for each client.
        * **config**: experiment configuration.
        * **run_num**: train number in the client.

    Code example:
    ``` 
    data = load_data(config, run_num=2)
    # data.train
    # data.test_normal
    # data.test_abnormal
    ```
    """
    config["run_num"] = run_num
    args = Arguments.from_config(config)
    # TODO: add functionality