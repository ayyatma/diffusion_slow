import pytest
import torch

from models.mobilevit_s_exits import MobileViTSWithExits


@pytest.mark.slow
def test_mobilevit_forward_contract(device, input_size):
    model = MobileViTSWithExits(num_classes=10, pretrained=False).to(device)
    model.eval()
    x = torch.randn(2, 3, input_size, input_size, device=device)

    with torch.no_grad():
        logits = model(x)

    assert isinstance(logits, list)
    assert len(logits) == 6, "expected exits 1-5 plus final classifier"
    for i, out in enumerate(logits, start=1):
        assert out.shape == (2, 10), f"bad shape at output {i}: {tuple(out.shape)}"
        assert torch.isfinite(out).all(), f"non-finite logits at output {i}"


@pytest.mark.slow
def test_mobilevit_feature_channels_match_exit_heads(input_size):
    model = MobileViTSWithExits(num_classes=10, pretrained=False)
    model.eval()
    x = torch.randn(1, 3, input_size, input_size)

    with torch.no_grad():
        features = model.get_intermediate_features(x)

    feature_channels = [feat.shape[1] for feat in features]
    exit_channels = []
    for exit_head in model.exits:
        if hasattr(exit_head, "conv"):
            exit_channels.append(exit_head.conv.in_channels)
        elif hasattr(exit_head, "norm"):
            exit_channels.append(exit_head.norm.normalized_shape[0])
        else:
            raise AssertionError(f"unknown exit head type: {type(exit_head).__name__}")

    assert feature_channels == exit_channels


@pytest.mark.slow
def test_mobilevit_final_output_matches_backbone_forward(device, input_size):
    model = MobileViTSWithExits(num_classes=10, pretrained=False).to(device)
    model.eval()
    x = torch.randn(2, 3, input_size, input_size, device=device)

    with torch.no_grad():
        all_logits = model(x)
        backbone_logits = model.backbone(x)

    assert torch.allclose(all_logits[-1], backbone_logits, atol=1e-6)
