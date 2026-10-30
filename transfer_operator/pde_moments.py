"""Empirical and exact moments with the registered 2048-sample accumulation order."""
from dataclasses import dataclass
import numpy as np

@dataclass
class TaskStats:
    """Uncentered sufficient statistics E_n[FF.T] and E_n[UF.T]."""
    n: int
    Cff: np.ndarray
    Cuf: np.ndarray


def empirical_task(F, U):
    """Form moments from column samples (d,n); no noise or simulation is added."""
    F, U = np.asarray(F), np.asarray(U)
    if F.ndim != 2 or U.shape != F.shape or F.shape[1] == 0:
        raise ValueError("F and U must have equal nonempty (d,n) shapes")
    n = F.shape[1]
    return TaskStats(n, F @ F.T / n, U @ F.T / n)


def population_task(model, theta, n=1):
    """Exact retained-Galerkin population moments (noise is centered)."""
    Cff = np.diag(model.q)
    return TaskStats(int(n), Cff, model.apply(model.source_h(theta), Cff))


def generate_task(model, n, theta, seed, profile='full', return_samples=False,
                  chunk_size=2048):
    """Generate a single registered task, with separate forcing/noise streams.

    The release defaults to the paper's full noise profile, including the
    constant mode. No experiment sweep is started by this function.
    """
    if int(n) != n or n <= 0:
        raise ValueError("n must be a positive integer")
    return _generate_advection_task(model, n, theta, seed, profile,
                                     return_samples, chunk_size)


def _generate_advection_task(model, n, theta, seed, profile='full',
                  return_samples=False, chunk_size=2048):
    """Independent sign forcing/noise in the registered full-noise model.

    Random streams are independent of chunk size and use a fixed mode order.
    Samples, when requested for validation, are returned as (d,n) arrays.
    """
    if profile != 'full':
        raise ValueError("The paper uses full noise, including the constant mode")
    sequence = seed if isinstance(seed, np.random.SeedSequence) else np.random.SeedSequence(seed)
    # Reconstructing a supplied sequence avoids consuming its spawn counter.
    sequence = np.random.SeedSequence(sequence.entropy, spawn_key=sequence.spawn_key)
    sf, se = sequence.spawn(2)
    rf, re = np.random.default_rng(sf), np.random.default_rng(se)
    sigma = model.noise_envelope.copy()
    Cff, Cef = np.zeros((model.d, model.d)), np.zeros((model.d, model.d))
    f_parts, e_parts = [], []
    for first in range(0, int(n), int(chunk_size)):
        size = min(int(chunk_size), int(n) - first)
        # Float uniforms consume one reproducible draw per coordinate.  The
        # int8 bit generator sampler can be shape/chunk dependent.
        F = (2.0 * (rf.random((size, model.d)) >= 0.5) - 1.0) * model.sqrt_q
        E = (2.0 * (re.random((size, model.d)) >= 0.5) - 1.0) * sigma
        Cff += F.T @ F
        Cef += E.T @ F
        if return_samples:
            f_parts.append(F.T)
            e_parts.append(E.T)
    Cff /= n
    Cuf = model.apply(model.source_h(theta), Cff) + Cef / n
    stats = TaskStats(int(n), Cff, Cuf)
    if not return_samples:
        return stats
    F, E = np.concatenate(f_parts, axis=1), np.concatenate(e_parts, axis=1)
    return stats, F, model.apply(model.source_h(theta), F) + E
