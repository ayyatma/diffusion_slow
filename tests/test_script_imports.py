def test_training_and_eval_entrypoints_import():
    from eval.check_checkpoint import check_checkpoint
    from eval.eval_clean import eval_clean
    from training.train_exits_stage0 import train_stage0
    from training.train_exits_stage1 import train_stage1
    from training.train_exits_stage2 import train_stage2

    assert callable(check_checkpoint)
    assert callable(eval_clean)
    assert callable(train_stage0)
    assert callable(train_stage1)
    assert callable(train_stage2)


def test_direct_script_help_entrypoints_work():
    import subprocess
    import sys

    scripts = [
        "training/train_exits_stage0.py",
        "training/train_exits_stage1.py",
        "training/train_exits_stage2.py",
        "eval/eval_clean.py",
        "eval/check_checkpoint.py",
    ]
    for script in scripts:
        result = subprocess.run(
            [sys.executable, script, "--help"],
            check=False,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
