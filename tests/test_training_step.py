import pytest
import torch
import torch.nn.functional as F

from models.mobilevit_s_exits import MobileViTSWithExits


def _require_grad(param):
    return param.requires_grad


@pytest.mark.slow
@pytest.mark.gpu
def test_joint_exit_training_step_has_finite_loss_and_gradients(device, input_size):
    model = MobileViTSWithExits(num_classes=10, pretrained=False).to(device)
    model.train()

    x = torch.randn(2, 3, input_size, input_size, device=device)
    y = torch.tensor([0, 1], device=device)
    weights = [0.1, 0.2, 0.4, 0.6, 0.8, 1.0]

    logits = model(x)
    loss = sum(weight * F.cross_entropy(out, y) for weight, out in zip(weights, logits))
    loss.backward()

    assert torch.isfinite(loss)

    for i, exit_head in enumerate(model.exits, start=1):
        for name, param in exit_head.named_parameters():
            assert param.grad is not None, f"exit {i} missing gradient for {name}"
            assert torch.isfinite(param.grad).all(), f"exit {i} non-finite gradient for {name}"

    trainable_without_grad = [
        name
        for name, param in model.named_parameters()
        if _require_grad(param) and param.grad is None
    ]
    assert not trainable_without_grad, "trainable params disconnected from loss: " + ", ".join(trainable_without_grad[:10])
