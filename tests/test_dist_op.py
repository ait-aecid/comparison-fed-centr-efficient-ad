
import op._dist_op as dist

import unittest


class DummyArgs:
    amount_clients = 2
    seed_number = 4

class DistTestCase(unittest.TestCase):
    def test_get_uniform_sample(self) -> None:
        unidist = dist.UniformDist(seed=4)
        sample = unidist.sample(5).tolist()
        
        self.assertListEqual([1., 1., 1., 1., 1.], sample)


    def test_call_uni(self) -> None:
        unidist = dist.UniformDist(seed=4)
        result = unidist(n_clients=4, plot_path=None)

        self.assertEqual(4, len(result))
        self.assertAlmostEqual(1., sum(result), delta=0.01)
        self.assertListEqual([0.25, 0.25, 0.25, 0.25], result)

    def test_split_by_dist_uniform(self) -> None:
        gen = dist.split_by_dist(DummyArgs(), n_idxs=10)

        self.assertTupleEqual((0, 5), next(gen))
        self.assertTupleEqual((5, 10), next(gen))

        args = DummyArgs()
        args.amount_clients = 4
        gen = dist.split_by_dist(args, n_idxs=20)

        self.assertTupleEqual((0, 5), next(gen))
        self.assertTupleEqual((5, 10), next(gen))
        self.assertTupleEqual((10, 15), next(gen))
        self.assertTupleEqual((15, 20), next(gen))