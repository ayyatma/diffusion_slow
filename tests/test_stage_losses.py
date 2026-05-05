import torch

from training.train_exits_stage2 import kd_loss


def test_kd_loss_is_finite_and_backpropagates():
    student_logits = torch.randn(4, 10, requires_grad=True)
    teacher_logits = torch.randn(4, 10)

    loss = kd_loss(student_logits, teacher_logits, temperature=4.0)
    loss.backward()

    assert torch.isfinite(loss)
    assert student_logits.grad is not None
    assert torch.isfinite(student_logits.grad).all()
