"""Exact coupled block DGP and population prediction risk.

The input F=(0,f) has independent Rademacher coordinates f_k=±sqrt(nu_k).
The response is U=(<a_j,f>+epsilon, Lambda f); epsilon is N(0,1).
Target a_0 and source a_j=a_0+Lambda sum_l theta_jl h_l are built from scalars.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import hashlib
import numpy as np

@dataclass
class BlockModel:
    name: str
    nu: np.ndarray          # (K,) forcing spectrum (= Gamma0 eigenvalues on the a-part)
    mu: np.ndarray          # (K,) Lambda diagonal
    a0: np.ndarray          # (K,) target parameter a
    h: np.ndarray           # (p, K) dictionary vectors h_l
    thetas: np.ndarray      # (m, p)
    sigma: float = 1.0
    beta: float = 1.0
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        self.nu, self.mu, self.a0 = (np.asarray(x, float) for x in (self.nu, self.mu, self.a0))
        self.h = np.atleast_2d(np.asarray(self.h, float))
        self.thetas = np.atleast_2d(np.asarray(self.thetas, float))

    @property
    def K(self):
        return self.nu.size

    @property
    def p(self):
        return self.h.shape[0]

    @property
    def m(self):
        return self.thetas.shape[0]

    def a_j(self, j: int) -> np.ndarray:
        return self.a0 + self.mu * (self.thetas[j] @ self.h)

    def a_of(self, theta) -> np.ndarray:
        return self.a0 + self.mu * (np.asarray(theta, float) @ self.h)

    def risk(self, c: float, a: np.ndarray) -> float:
        return float(np.sum(self.nu * (a - self.a0) ** 2) + (c - 1.0) ** 2 * np.sum(self.mu ** 2 * self.nu))

    def risk_a_only(self, a: np.ndarray) -> float:
        return float(np.sum(self.nu * (a - self.a0) ** 2))

    def eff_dim(self, lam: float) -> float:
        return float(np.sum(self.nu / (self.nu + lam)))

    def norm_H2_A0(self) -> float:
        return float(1.0 + np.sum(self.a0 ** 2))

    def T_h(self, hvec) -> np.ndarray:
        """T_h = A0 (h.D) A0 = T_{0, -Lambda h_h}: returns the a-part; its Gamma0-norm^2 is sum nu (Lambda h_h)^2."""
        return -self.mu * (np.asarray(hvec, float) @ self.h)

    def with_thetas(self, thetas) -> "BlockModel":
        return BlockModel(self.name, self.nu, self.mu, self.a0, self.h, np.atleast_2d(np.asarray(thetas, float)),
                          self.sigma, self.beta, dict(self.meta))


ARRAY_NAMES = ("nu", "mu", "a0", "h", "thetas")


def model_checksums(model):
    """SHA-256 of little-endian float64 arrays in the original DGP order."""
    payloads = [np.ascontiguousarray(getattr(model, key), dtype="<f8").tobytes()
                for key in ARRAY_NAMES]
    return dict(model_arrays_sha256=hashlib.sha256(b"".join(payloads)).hexdigest(),
                dictionary_sha256=hashlib.sha256(payloads[3]).hexdigest())


def build_model(spec):
    """Rebuild coupled model arrays, preserving the archived operation order.

    In particular, normalize k**(-target_exponent) with np.linalg.norm;
    replacing its norm with a mathematically equivalent power sum changes bits.
    The perturbation weights use ((nu+eta)**(-.5)*nu*mu) before multiplying h.T.
    """
    scalars, dictionary, perturbation = (spec[key] for key in
                                        ("model", "dictionary", "perturbation"))
    if scalars["family"] != "coupled" or dictionary["layout"] != "odd_even":
        raise ValueError("Supported closed-form model: coupled with odd_even dictionary")
    size, cutoff = scalars["K"], dictionary["cutoff"]
    if (isinstance(size, bool) or int(size) != size or size < 2
            or isinstance(cutoff, bool) or int(cutoff) != cutoff or not 2 <= cutoff <= size):
        raise ValueError("K and dictionary cutoff must be integers with 2 <= cutoff <= K")
    if perturbation["balance"] != "w1/w2":
        raise ValueError("Unsupported perturbation balance")
    k = np.arange(1, int(size) + 1, dtype=float)
    nu = k ** (-float(scalars["nu_exponent"]))
    mu = k ** (-float(scalars["mu_exponent"]))
    v = k ** (-float(scalars["target_exponent"]))
    a0 = float(scalars["R"]) * (v / np.linalg.norm(v))
    h = np.zeros((2, int(size)))
    for row, parity in enumerate((1, 0)):
        support = (k <= cutoff) & (k % 2 == parity)
        h[row, support] = k[support] ** (-float(dictionary["exponent"]))
        h[row] /= np.linalg.norm(h[row])
    g = ((nu + float(perturbation["reference_eta"])) ** (-.5) * nu * mu)[:, None] * h.T
    w = np.sum(g ** 2, axis=0)
    tau1 = float(perturbation["tau1"])
    thetas = np.diag([tau1, tau1 * w[0] / w[1]])
    model = BlockModel(scalars.get("name", "block"), nu, mu, a0, h, thetas,
                       sigma=float(scalars["sigma"]), beta=float(scalars["beta"]),
                       meta=dict(K_D=int(cutoff), h_layout="low_frequency_odd_even"))
    if not all(np.isfinite(getattr(model, key)).all() for key in ARRAY_NAMES):
        raise ValueError("Model arrays must be finite")
    for key, digest in model_checksums(model).items():
        if key in spec.get("checksums", {}) and digest != spec["checksums"][key]:
            raise ValueError(f"Rebuilt {key} checksum mismatch")
    return model


def load_design(path):
    """Read an experiment's config.yaml and expand its registered study design."""
    from .config import ExperimentConfig
    return ExperimentConfig.from_directory(path).design


def load_model(path):
    """Reconstruct and checksum an experiment's config.yaml model."""
    from .config import ExperimentConfig
    return ExperimentConfig.from_directory(path).model
