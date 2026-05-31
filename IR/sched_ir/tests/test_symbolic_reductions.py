from IR.sched_ir.correctness.reductions import reduce_scheduled_values


def fake_add(left, right):
    return f"({left}+{right})"


def test_spatial_reduction_keeps_single_symbolic_value():
    result = reduce_scheduled_values(
        ["whole-tree"],
        mode="spatial",
        reducer=fake_add,
    )

    assert result == "whole-tree"


def test_hybrid_reduction_combines_partial_temporal_values_in_step_order():
    result = reduce_scheduled_values(
        ["partial0", "partial1", "partial2"],
        mode="hybrid",
        reducer=fake_add,
    )

    assert result == "((partial0+partial1)+partial2)"


def test_temporal_accumulate_uses_same_accumulation_semantics_as_hybrid():
    result = reduce_scheduled_values(
        ["lane0", "lane1"],
        mode="temporal_accumulate",
        reducer=fake_add,
    )

    assert result == "(lane0+lane1)"
