import torch
import torch.nn.functional as F

def prediction_entropy(logits: torch.Tensor) -> torch.Tensor:
    """
    Shannon entropy of softmax distribution.
    logits: (B, num_classes)
    returns: (B,) entropy values in nats
    """
    probs = F.softmax(logits, dim=-1)
    log_probs = F.log_softmax(logits, dim=-1)
    entropy = -torch.sum(probs * log_probs, dim=-1)
    # Note: Can be normalized by log(num_classes) if desired.
    return entropy

def prediction_entropy_normalized(logits: torch.Tensor) -> torch.Tensor:
    """
    Normalized Shannon entropy (values in [0, 1]).
    """
    entropy = prediction_entropy(logits)
    num_classes = logits.shape[-1]
    return entropy / torch.log(torch.tensor(num_classes, dtype=logits.dtype, device=logits.device))

def max_confidence(logits: torch.Tensor) -> torch.Tensor:
    """
    Maximum softmax probability (used for exit decisions).
    logits: (B, num_classes)
    returns: (B,) max prob values
    """
    probs = F.softmax(logits, dim=-1)
    return probs.max(dim=-1).values

def exit_decision(logits: torch.Tensor, threshold: float) -> torch.Tensor:
    """
    Returns True for samples that should exit (confident enough).
    logits: (B, num_classes)
    threshold: float
    returns: (B,) boolean mask
    """
    return max_confidence(logits) >= threshold
