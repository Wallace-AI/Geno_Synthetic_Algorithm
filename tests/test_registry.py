import pytest

from gsa.experiments.registry import (
    Registry,
    get_algorithm,
    get_benchmark,
    register_algorithm,
    register_benchmark,
)


def test_register_and_retrieve():
    @register_algorithm("DUMMY_ALGO")
    def factory(config):
        return f"algo:{config}"

    assert get_algorithm("DUMMY_ALGO")("x") == "algo:x"


def test_duplicate_registration_raises():
    @register_benchmark("dummy_bench")
    def f(config):
        return "bench"

    with pytest.raises(ValueError, match="already registered"):
        @register_benchmark("dummy_bench")
        def g(config):
            return "bench2"


def test_unknown_lookup_raises():
    with pytest.raises(KeyError):
        get_algorithm("DOES_NOT_EXIST")
