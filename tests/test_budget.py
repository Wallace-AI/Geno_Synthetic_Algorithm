import pytest
from gsa.experiments.budget import EvaluationBudget, BudgetExceeded


def test_budget_consumes_units():
    b = EvaluationBudget(total=100)
    b.consume(10)
    assert b.consumed == 10
    assert b.remaining == 90


def test_budget_exhausted_raises():
    b = EvaluationBudget(total=10)
    b.consume(10)
    with pytest.raises(BudgetExceeded):
        b.consume(1)


def test_budget_partial_overconsume_clipped_with_strict_false():
    b = EvaluationBudget(total=10, strict=False)
    b.consume(5)
    consumed = b.consume(20)  # only 5 left, returns 5
    assert consumed == 5
    assert b.consumed == 10


def test_budget_check_before_consume():
    b = EvaluationBudget(total=10)
    assert b.has(5)
    b.consume(8)
    assert not b.has(5)
    assert b.has(2)
