import numpy as np
import pytest
from gsa.benchmarks.base import Problem
from gsa.experiments.budget import EvaluationBudget


def test_problem_is_abstract():
    with pytest.raises(TypeError):
        Problem()


def test_subclass_must_implement_evaluate_specs_optimum():
    class Bad(Problem):
        pass

    with pytest.raises(TypeError):
        Bad(budget=EvaluationBudget(100), seed=0)
