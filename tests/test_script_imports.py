def test_training_and_eval_entrypoints_import():
    from eval.eval_clean import eval_clean
    from training.train_exits_stage0 import train_stage0
    from training.train_exits_stage1 import train_stage1
    from training.train_exits_stage2 import train_stage2

    assert callable(eval_clean)
    assert callable(train_stage0)
    assert callable(train_stage1)
    assert callable(train_stage2)
