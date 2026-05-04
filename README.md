# gsa-experiments

Research-grade experiment battery for the **Geno-Synthetic Algorithm (GSA)**, a
type-factored coevolutionary optimization framework. This repository
implements GSA, baseline algorithms (CMA-ES, GA, NSGA-II, BO, etc.), and a
benchmark suite (BBOB, ZDT/DTLZ, CEC, hyperparameter tuning, NK landscapes,
ALife) used to empirically characterize GSA's behavior, scaling, and limits
relative to established methods.

## Quickstart

```bash
python -m venv .venv
# Windows:
.venv/Scripts/activate
# Unix:
# source .venv/bin/activate

pip install -e .[test]

# Smoke test: package imports and reports its version
python -c "import gsa; print(gsa.__version__)"
# Expected: 0.1.0

pytest
```

## Layout

- `src/gsa/` — core, baselines, benchmarks, experiments, analysis
- `configs/` — experiment YAML configs
- `scripts/` — entry points for runs and analysis
- `tests/` — unit and integration tests
- `results/` — raw/processed run artifacts, figures, tables, reports
- `docs/` — design specs, plans, notes
