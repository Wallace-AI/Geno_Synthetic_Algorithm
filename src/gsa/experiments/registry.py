"""Decorator-based registries for algorithms and benchmarks.

Algorithms and benchmarks register themselves at module-import time via
`@register_algorithm("KEY")` / `@register_benchmark("key")` decorators.
The experiment runner looks them up by string key. Module-level globals
are intentional: registry state is process-wide and persists for the
lifetime of the run.
"""
from __future__ import annotations

from typing import Callable


class Registry:
    def __init__(self, name: str) -> None:
        self.name = name
        self._items: dict[str, Callable] = {}

    def register(self, key: str) -> Callable[[Callable], Callable]:
        def decorator(fn: Callable) -> Callable:
            if key in self._items:
                raise ValueError(f"{self.name} key {key!r} already registered")
            self._items[key] = fn
            return fn
        return decorator

    def get(self, key: str) -> Callable:
        if key not in self._items:
            raise KeyError(
                f"{self.name} key {key!r} not found; known: {sorted(self._items)}"
            )
        return self._items[key]

    def keys(self) -> list[str]:
        return sorted(self._items)


_ALGORITHMS = Registry("algorithm")
_BENCHMARKS = Registry("benchmark")


def register_algorithm(key: str) -> Callable[[Callable], Callable]:
    return _ALGORITHMS.register(key)


def register_benchmark(key: str) -> Callable[[Callable], Callable]:
    return _BENCHMARKS.register(key)


def get_algorithm(key: str) -> Callable:
    return _ALGORITHMS.get(key)


def get_benchmark(key: str) -> Callable:
    return _BENCHMARKS.get(key)


def list_algorithms() -> list[str]:
    return _ALGORITHMS.keys()


def list_benchmarks() -> list[str]:
    return _BENCHMARKS.keys()
