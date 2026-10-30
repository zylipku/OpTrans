"""Shared transfer-operator models, estimators, simulation and study configuration."""
import os
# A single BLAS thread preserves the reference reduction order.
for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_name] = "1"

from .config import ExperimentConfig
from .model import BlockModel, load_design, load_model
from .estimators import Candidate, FoldFit, SourceBank, candidate_registry, iteration_path
from .simulation import development_world, run_trajectory
from .development import run_development, select_candidates

__all__ = ["ExperimentConfig", "BlockModel", "load_design", "load_model", "Candidate",
           "FoldFit", "SourceBank", "candidate_registry", "iteration_path",
           "development_world", "run_trajectory", "run_development", "select_candidates"]

# PDE stages extend the package without changing the coupled numerical backend.
from .pde_models import PDEModel, AdvectionDiffusionModel, FourierModel
from .pde_moments import TaskStats, empirical_task, population_task, generate_task
from .pde_pipeline import Tuning, TransferFit, frozen_tuning, pilot, recalibration_round, fit_transfer
from .stages import CoupledProblem, PDEProblem, StagePath, fit_stages

__all__ += ["PDEModel", "AdvectionDiffusionModel", "FourierModel",
            "TaskStats", "empirical_task", "population_task", "generate_task",
            "Tuning", "TransferFit", "frozen_tuning", "pilot", "recalibration_round",
            "fit_transfer", "CoupledProblem", "PDEProblem", "StagePath", "fit_stages"]

from .pde_config import PDECell, PDEStudy
__all__ += ["PDECell", "PDEStudy"]
