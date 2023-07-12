from compatibility.dataclass_shim import dataclass
from pathlib import Path


@dataclass
class ProjectPaths:
    datasets: Path
    mlruns: Path
    containers: Path
    slurm: Path
    matplotlib_cache: Path


@dataclass
class Container:
    singularity_container_name: str


@dataclass
class Logging:
    # Allowed values: 'sqlite' and 'filesystem'. For comp. with python<3.8, don't use typing.Literal
    mlflow_backend: str
