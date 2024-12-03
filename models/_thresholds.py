from metrics import apply_metrics
import torch
from tqdm import tqdm 

import typing as t


def apply_threshold(
    score: t.List[float] | torch.Tensor, threshold: float, 
) -> t.List[int]:
    return (torch.Tensor(score) >= threshold).int().detach().tolist()


def supervised_threshold_selection(
    score_normal: t.List[float], 
    score_abnormal: t.List[float],
    metric: str = "f1",
) -> float:

    normal = torch.Tensor(score_normal)
    abnormal = torch.Tensor(score_abnormal)
    all_values = torch.cat((normal, abnormal))
    
    thresholds = torch.linspace(
        start=torch.min(all_values),
        end=torch.max(all_values),
        steps=1000,
    )
    results = []
    for thres in tqdm(thresholds):
        metrics = apply_metrics(
            pred_normal=apply_threshold(normal, threshold=thres),
            pred_abnormal=apply_threshold(abnormal, threshold=thres),
        )
        results.append(getattr(metrics, metric))

    return thresholds[torch.argmax(torch.Tensor(results))].detach().tolist()