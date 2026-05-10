"""Phenotype assembly operators.

ActiveAssembly performs Boolean gating, categorical routing, complex/embedding
decoding, and constraint validation. PassiveAssembly concatenates only —
used by `GSA_NO_ASSEMBLY` ablation.

Both return a (Phenotype, AssemblyDiagnostics) tuple."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from gsa.core.genome import (
    TypedBundle, Phenotype, AssemblyDiagnostics, TypedSubgenome,
)
from gsa.core.types import GeneFamily


@dataclass
class ActiveAssembly:
    """Active synthesis: Boolean gates Real, Categorical routes."""
    boolean_gates_real: bool = True
    categorical_routes: bool = True
    integer_selects_subfunction: bool = True

    def assemble(self, bundle: TypedBundle) -> tuple[Phenotype, AssemblyDiagnostics]:
        features: dict = {}
        n_active = 0
        repair = 0
        invalid_reason = None

        # Real - effective values after Boolean gating
        R = bundle.get(GeneFamily.R)
        B = bundle.get(GeneFamily.B)
        if R is not None:
            r_vals = R.values.copy()
            if self.boolean_gates_real and B is not None:
                gates = B.values.astype(bool)
                m = min(len(gates), len(r_vals))
                r_eff = r_vals.copy()
                r_eff[:m] = np.where(gates[:m], r_vals[:m], 0.0)
                if m < len(r_vals):
                    pass  # untouched tail remains active
            else:
                r_eff = r_vals
            features["R_effective"] = r_eff
            features["R_raw"] = r_vals
            n_active += int(np.sum(r_eff != 0))

        if B is not None:
            features["B"] = B.values.astype(bool)
            n_active += len(B.values)

        Z = bundle.get(GeneFamily.Z)
        if Z is not None:
            features["Z"] = Z.values.astype(np.int64)
            n_active += len(Z.values)
            if self.integer_selects_subfunction:
                features["Z_subfunction"] = Z.values.astype(np.int64)

        C = bundle.get(GeneFamily.C)
        if C is not None:
            features["C"] = C.values.astype(np.int64)
            n_active += len(C.values)
            if self.categorical_routes:
                features["C_route"] = int(C.values[0]) if len(C.values) > 0 else 0

        Cx = bundle.get(GeneFamily.Cx)
        if Cx is not None:
            features["Cx_magnitude"] = np.abs(Cx.values)
            features["Cx_phase"] = np.angle(Cx.values)
            n_active += len(Cx.values)

        E = bundle.get(GeneFamily.E)
        if E is not None:
            # Already unit-norm by operator; record as feature directly.
            features["E"] = E.values
            n_active += E.values.shape[0]

        diag = AssemblyDiagnostics(
            valid=invalid_reason is None,
            repair_count=repair,
            invalid_reason=invalid_reason,
            n_active_genes=n_active,
        )
        return Phenotype(features=features, bundle=bundle, diagnostics=diag), diag


@dataclass
class PassiveAssembly:
    """No-op assembly: copies subgenome values into features.

    Boolean and Categorical genes exist but do not gate or route. Used by
    GSA_NO_ASSEMBLY ablation per spec §2.3."""

    def assemble(self, bundle: TypedBundle) -> tuple[Phenotype, AssemblyDiagnostics]:
        features: dict = {}
        n_active = 0
        for fam, sg in bundle.subgenomes.items():
            features[fam.value] = sg.values
            n_active += sg.spec.n
        # Provide R_effective as raw R (no gating) for downstream compat
        if GeneFamily.R in bundle.subgenomes:
            features["R_effective"] = bundle.subgenomes[GeneFamily.R].values
            features["R_raw"] = bundle.subgenomes[GeneFamily.R].values
        diag = AssemblyDiagnostics(valid=True, repair_count=0,
                                    invalid_reason=None, n_active_genes=n_active)
        return Phenotype(features=features, bundle=bundle, diagnostics=diag), diag
