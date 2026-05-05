import math

import torch

from attacks.utils.entropy import (
    exit_decision,
    max_confidence,
    prediction_entropy,
    prediction_entropy_normalized,
)


def test_entropy_confidence_and_exit_decision_are_correct():
    logits = torch.tensor([[10.0, 0.0, -10.0], [0.0, 0.0, 0.0]])

    entropy = prediction_entropy(logits)
    norm_entropy = prediction_entropy_normalized(logits)
    confidence = max_confidence(logits)

    assert entropy.shape == (2,)
    assert confidence.shape == (2,)
    assert entropy[0] < entropy[1]
    assert torch.isclose(entropy[1], torch.tensor(math.log(3.0)), atol=1e-6)
    assert torch.all(norm_entropy >= 0)
    assert torch.all(norm_entropy <= 1 + 1e-6)

    assert exit_decision(logits, threshold=0.90).tolist() == [True, False]
    assert exit_decision(logits, threshold=1 / 3).tolist() == [True, True]


def test_entropy_handles_batched_large_logits_without_nan():
    logits = torch.tensor([[1000.0, -1000.0], [-1000.0, 1000.0]])

    entropy = prediction_entropy(logits)
    confidence = max_confidence(logits)

    assert torch.isfinite(entropy).all()
    assert torch.isfinite(confidence).all()
    assert torch.allclose(confidence, torch.ones(2))
