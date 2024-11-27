
import models.vector_count as vc
import torch

import unittest


class TensorOperationTestCase(unittest.TestCase):
    def test_generate_vector(self) -> None:
        x = [[1, 4, 2, 1, 2], [0, 0, 0, 3]]
        n_elemts = 5

        vector = vc.generate_vector(x, n_elemts=n_elemts).detach().tolist()
        self.assertEqual(len(vector), 2)
        self.assertListEqual(vector[0], [0, 2, 2, 0, 1])
        self.assertListEqual(vector[1], [3, 0, 0, 1, 0])

    def test_make_equal(self) -> None:
        x1 = torch.ones((4, 3))
        x2 = torch.ones((2, 5))
        new_x1, new_x2 = vc.convert_same_shape(x1, x2)

        self.assertEqual(new_x1.shape[1], new_x2.shape[1])
        self.assertListEqual(
            [[1., 1., 1., 0., 0.] for _ in range(4)], new_x1.detach().tolist()
        )

        x2 = torch.ones((4, 3))
        x1 = torch.ones((2, 5))
        new_x1, new_x2 = vc.convert_same_shape(x1, x2)

        self.assertEqual(new_x1.shape[1], new_x2.shape[1])
        self.assertListEqual(
            [[1., 1., 1., 0., 0.] for _ in range(4)], new_x2.detach().tolist()
        )

    def test_max_element_wise(self) -> None:
        X1 = torch.Tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
        x2 = torch.Tensor([1, -1, 5, 10])
        new_x = vc.max_elemnt_wise(X1=X1, x2=x2).detach().tolist()

        self.assertListEqual(new_x, [[1, 2, 5, 10], [5, 6, 7, 10]])


class CountVectorTestCase(unittest.TestCase):
    def test_get_weights(self) -> None:
        count_vector = vc.CountVector()
        count_vector.vectors = vc.torch.Tensor([1, 2, 3, 4])
        
        self.assertTrue(isinstance(count_vector.get_weights(), list))
        self.assertListEqual(
            count_vector.get_weights(), [1, 2, 3, 4]
        )

    def test_set_weights(self) -> None:
        count_vector = vc.CountVector()
        count_vector.set_weights([1, 2, 3, 4]) 
        
        self.assertTrue(isinstance(count_vector.get_weights(), list))
        self.assertListEqual(
            count_vector.get_weights(), [1, 2, 3, 4]
        )

    def test_fit(self) -> None: 
        x = [[1, 4, 2, 1, 2], [0, 0, 0, 3], [0, 0, 0, 3], [4, 0], [0, 4]]
        count_vector = vc.CountVector()
        
        self.assertEqual(3, count_vector.fit(x))
        vectors = count_vector.get_weights()
        self.assertTrue([0, 2, 2, 0, 1] in vectors)
        self.assertTrue([3, 0, 0, 1, 0] in vectors)
        self.assertTrue([1, 0, 0, 0, 1] in vectors)

    def test_predict(self) -> None:
        x = [[1, 4, 2, 1, 2], [0, 0, 0, 3], [0, 0, 0, 3], [4, 0], [0, 4]]
        x_test = [[0, 0, 3, 0], [4, 5, 5, 5, 5, 7, 8], [0, 4]]
        count_vector = vc.CountVector()
        count_vector.fit(x)

        pred = count_vector.predict(x_test)
        self.assertEqual(len(pred), 3)
        for p, r in zip(pred, [0, 0.2927, 0]):
            self.assertAlmostEqual(p, r, delta=1e-3)

    def test_update_strategy(self) -> None:
        count_vector = vc.CountVector()
        client_weights = [
            [[1, 2, 0, 0, 0], [0, 0, 1, 0, 1]], [[1, 0], [2, 1]], [[1]], []
        ]
        weights = vc.update_strategy(count_vector, clients_weights=client_weights)
        expected = [
            [1, 2, 0, 0, 0], [0, 0, 1, 0, 1], [1, 0, 0, 0, 0], [2, 1, 0, 0, 0]
        ]

        self.assertEqual(len(weights), len(expected))
        for w in weights:
            self.assertTrue(w in expected)

