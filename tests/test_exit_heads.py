import torch

from models.exit_heads import GlobalAggregationHead, LocalPerceptionHead


def test_local_perception_head_output_shape_and_gradients():
    head = LocalPerceptionHead(in_channels=32, num_classes=10)
    x = torch.randn(4, 32, 32, 32, requires_grad=True)

    logits = head(x)
    loss = logits.mean()
    loss.backward()

    assert logits.shape == (4, 10)
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    for name, param in head.named_parameters():
        assert param.grad is not None, f"missing gradient for {name}"
        assert torch.isfinite(param.grad).all(), f"non-finite gradient for {name}"


def test_global_aggregation_head_output_shape_and_gradients():
    head = GlobalAggregationHead(in_channels=96, num_classes=10)
    x = torch.randn(4, 96, 16, 16, requires_grad=True)

    logits = head(x)
    loss = logits.mean()
    loss.backward()

    assert logits.shape == (4, 10)
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    for name, param in head.named_parameters():
        assert param.grad is not None, f"missing gradient for {name}"
        assert torch.isfinite(param.grad).all(), f"non-finite gradient for {name}"
