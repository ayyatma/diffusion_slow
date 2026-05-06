import torch

from training.common import load_matching_state_dict


def test_load_matching_state_dict_can_exclude_prefixes(tmp_path):
    source = torch.nn.Sequential(torch.nn.Linear(2, 3), torch.nn.Linear(3, 1))
    target = torch.nn.Sequential(torch.nn.Linear(2, 3), torch.nn.Linear(3, 1))
    path = tmp_path / "state.pt"
    torch.save(source.state_dict(), path)

    report = load_matching_state_dict(target, path, torch.device("cpu"), exclude_prefixes=("1.",))

    assert report["loaded_keys"] == ["0.bias", "0.weight"]
    assert report["skipped_keys"] == ["1.bias", "1.weight"]
    assert "1.weight" in report["missing_keys"]
    assert "1.bias" in report["missing_keys"]
