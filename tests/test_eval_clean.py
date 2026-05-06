import torch

from eval.eval_clean import calibrate_thresholds, evaluate_accuracy
from attacks.utils.entropy import max_confidence


class FixedLogitModel(torch.nn.Module):
    def __init__(self, outputs):
        super().__init__()
        self.outputs = outputs

    def forward(self, x):
        batch_size = x.shape[0]
        return [out[:batch_size].to(x.device) for out in self.outputs]


def _loader(batch_size=4):
    x = torch.zeros(batch_size, 3, 8, 8)
    y = torch.arange(batch_size) % 10
    return [(x, y)]


def test_calibrate_thresholds_is_sequential():
    # Exit 1 confidently exits samples 0 and 1. Exit 2 should calibrate only
    # from samples 2 and 3, so its threshold should be based on 0.9/0.8 rather
    # than the high-confidence samples already removed by exit 1.
    exit1 = torch.tensor([
        [10.0, 0.0],
        [9.0, 0.0],
        [0.2, 0.0],
        [0.1, 0.0],
    ])
    exit2 = torch.tensor([
        [8.0, 0.0],
        [7.0, 0.0],
        [2.2, 0.0],
        [1.4, 0.0],
    ])
    later = torch.zeros(4, 2)
    model = FixedLogitModel([exit1, exit2, later, later, later, later])

    thresholds = calibrate_thresholds(
        model,
        _loader(),
        torch.device("cpu"),
        target_exit_rate=0.50,
    )

    assert len(thresholds) == 5
    assert 0.79 < thresholds[1] < 0.92

    independent_exit2_threshold = torch.quantile(max_confidence(exit2), 0.50).item()
    assert thresholds[1] < independent_exit2_threshold


def test_evaluate_accuracy_returns_six_outputs():
    outputs = []
    for _ in range(6):
        logits = torch.zeros(4, 10)
        logits[torch.arange(4), torch.arange(4)] = 1.0
        outputs.append(logits)
    model = FixedLogitModel(outputs)

    accuracies = evaluate_accuracy(model, _loader(), torch.device("cpu"))

    assert accuracies == [1.0] * 6
