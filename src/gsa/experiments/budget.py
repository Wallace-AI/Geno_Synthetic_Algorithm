"""Evaluation budget: counts fitness-function calls.

Per spec §2.4 budget-counting rule: each fitness evaluation consumes 1 unit;
ensemble-credit consumes 5 per credit update; marginal-credit consumes 2.
The Problem ABC delegates increment to this object. Algorithms cannot
circumvent this: budget is checked at evaluation time."""
from __future__ import annotations


class BudgetExceeded(RuntimeError):
    """Raised when consume() would exceed the total in strict mode."""


class EvaluationBudget:
    def __init__(self, total: int, strict: bool = True) -> None:
        self.total = total
        self.strict = strict
        self.consumed = 0

    @property
    def remaining(self) -> int:
        return max(0, self.total - self.consumed)

    def has(self, n: int) -> bool:
        return self.remaining >= n

    def consume(self, n: int = 1) -> int:
        """Consume n units. Returns actual consumed (may be less if non-strict)."""
        if self.consumed + n > self.total:
            if self.strict:
                raise BudgetExceeded(
                    f"requested {n}, only {self.remaining} remaining of {self.total}"
                )
            actual = self.remaining
            self.consumed = self.total
            return actual
        self.consumed += n
        return n
