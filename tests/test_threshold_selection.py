from models._thresholds import supervised_threshold_selection, apply_threshold
from metrics import apply_metrics

import unittest


class ThresholdTestCase(unittest.TestCase):
    def setUp(self) -> None: 
        self.normal = [0.4, 0.6, 0.3, 0.7, 2.0]
        self.abnormal = [1.2, 0.9, 1.1, 0.01]

    def test_apply_threshold(self) -> None:
        score = [1, 2, 3, 4, 5, 6]

        self.assertListEqual(
            apply_threshold(score=score, threshold=0.7),
            [1, 1, 1, 1, 1, 1]
        )
        self.assertListEqual(
            apply_threshold(score=score, threshold=7),
            [0, 0, 0, 0, 0, 0]
        )
    
    def test_best_f1(self) -> None:
        thres = supervised_threshold_selection(
            score_normal=self.normal, score_abnormal=self.abnormal, metric="f1"
        )
        pred_normal = apply_threshold(self.normal, threshold=thres)
        pred_abnormal = apply_threshold(self.abnormal, threshold=thres)
        
        result = apply_metrics(
            pred_normal=pred_normal, pred_abnormal=pred_abnormal
        )
        self.assertEqual(result.f1, 0.75)

    def test_best_recall(self) -> None:
        thres = supervised_threshold_selection(
            score_normal=self.normal, score_abnormal=self.abnormal, metric="recall"
        )
        pred_normal = apply_threshold(self.normal, threshold=thres)
        pred_abnormal = apply_threshold(self.abnormal, threshold=thres)
        
        result = apply_metrics(
            pred_normal=pred_normal, pred_abnormal=pred_abnormal
        )
        self.assertEqual(result.recall, 1.)

    def test_best_precision(self) -> None:
        thres = supervised_threshold_selection(
            score_normal=self.normal, score_abnormal=self.abnormal, metric="precision"
        )
        pred_normal = apply_threshold(self.normal, threshold=thres)
        pred_abnormal = apply_threshold(self.abnormal, threshold=thres)
        
        result = apply_metrics(
            pred_normal=pred_normal, pred_abnormal=pred_abnormal
        )
        self.assertEqual(result.precision, 0.75)