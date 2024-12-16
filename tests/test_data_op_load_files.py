
import op._data_op as read_

import unittest


args = read_.Arguments.from_config({
    "dataset_path": "datasets/BGL",
    "amount_clients": 10,
    "seed_number": 7,
    "train_per": 0.1,
    "dist_method": "Uniform"
})
data = read_.read_files(args)


class ReadFilesTestCase(unittest.TestCase):
    def test_columns_names(self) -> None:        
        self.assertListEqual(["Normal", "Abnormal"], list(data.keys()))
        self.assertListEqual(
            ["ID", "Event_seq"], list(data["Normal"].keys())
        )
        self.assertListEqual(
            ["ID", "Event_seq"], list(data["Abnormal"].keys())
        )

    def test_format(self) -> None:
        self.assertTrue(
            isinstance(data["Normal"]["Event_seq"].iloc[0], list)
        )
        self.assertTrue(
            isinstance(data["Abnormal"]["Event_seq"].iloc[0], list)
        )

    def test_random_seed(self) -> None:
        for _ in range(100):
            read_.Random.do_seed(2)
            self.assertListEqual(
                [0, 1, 1], 
                read_.np.random.choice([0, 1], size=3).tolist()
            )

    def test_split_train_test(self) -> None:
        result = read_.split_train_test(
            args=args, normal=data["Normal"], num_run=0
        )
        result2 = read_.split_train_test(
            args=args, normal=data["Normal"], num_run=0
        )

        self.assertAlmostEqual(
            len(result["train"]), len(data["Normal"]) * 0.1, delta=10
        )
        self.assertAlmostEqual(
            len(result["test_normal"]), len(data["Normal"]) * 0.9, delta=10
        )
        self.assertTrue(result["train"].equals(result2["train"]))
        self.assertTrue(result["test_normal"].equals(result2["test_normal"]))

    def test_split_train_test_different_runs(self) -> None:
        result = read_.split_train_test(
            args=args, normal=data["Normal"], num_run=1
        )
        result2 = read_.split_train_test(
            args=args, normal=data["Normal"], num_run=2
        )

        self.assertFalse(result["train"].equals(result2["train"]))
        self.assertFalse(result["test_normal"].equals(result2["test_normal"]))

    def test_split_clients(self) -> None:
        splits = read_.split_in_clients(
            args=args, normal=data["Normal"],
        ) 
        splits2 = read_.split_in_clients(
            args=args, normal=data["Normal"],
        )

        self.assertEqual(args.amount_clients, len(splits))
        self.assertEqual(args.amount_clients, len(splits2))
        self.assertFalse(splits[0].equals(splits[1]))
        m = 0
        for i in range(len(splits)):
            m += len(splits[i])
            self.assertTrue(splits[i].equals(splits2[i]))
        self.assertAlmostEqual(
            m, len(data["Normal"]), delta=10
        )