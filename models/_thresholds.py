from metrics import apply_metrics
from op.aux import Color

from tqdm import tqdm 
import typing as t
import torch


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

    final_thres = thresholds[torch.argmax(torch.Tensor(results))].detach().tolist()
    print(Color.yellow(f"Final threshold {final_thres}"))

    return final_thres


# %% Thresholds setup for the different experiments
class Thresholds:
    events: float | None = None
    length: float | None = None
    ecvc: float | None = None 
    gram2: float | None = None 
    gram3: float | None = None
    edit: float | None = None


class ThresHDFS(Thresholds):
    ecvc: float | None = 0.00228
    gram2: float | None = 0.0224
    gram3: float | None = 0.0147
    edit: float | None = 0.18498


class ThresBGL(Thresholds):
    ecvc: float | None = 0.0018
    gram2: float | None = 0.0
    gram3: float | None = 0.0
    edit: float | None = 0.17084