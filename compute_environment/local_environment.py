#!/usr/bin/env python

from pathlib import Path
from compute_environment.compute_environment_config import ProjectPaths, Container, Logging

ROOT_DIR = Path(__file__).parents[1].absolute()

PATHS = ProjectPaths(
    datasets=ROOT_DIR / Path("datasets"),
    mlruns=ROOT_DIR / Path("mlruns"),
    containers=ROOT_DIR / Path("containers"),
    slurm=ROOT_DIR / Path("slurm"),
    matplotlib_cache=ROOT_DIR / Path("mpl_cache"),
)

CONTAINER = Container(singularity_container_name="heal_swin_container.sif")

LOGGING = Logging(mlflow_backend="filesystem")
