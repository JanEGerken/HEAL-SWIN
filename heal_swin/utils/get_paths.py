from pathlib import Path

from compute_environment import compute_environment


def get_datasets_path(subfolder):
    """Get datasets path from current environment"""
    return str((compute_environment.PATHS.datasets / subfolder).absolute())


def get_syn_datasets_path():
    woodscape_path = get_datasets_path("synwoodscape")
    return woodscape_path


def get_mlruns_path():
    """Get mlruns path from MLRUNSPATH environment variable"""
    return str(compute_environment.PATHS.mlruns.absolute())


def get_mlflow_db_path():
    return str(Path(get_mlruns_path()) / "mlflow.db")


def get_tracking_server_file_path():
    return str(Path(get_mlruns_path()) / "tracking_server_running.json")


def get_container_path():
    return str(compute_environment.PATHS.containers.absolute())


def get_slurm_path():
    return str(compute_environment.PATHS.slurm.absolute())


def get_mpl_cache_path():
    return str(compute_environment.PATHS.matplotlib_cache.absolute())


def get_base_path():
    """Get base path of the repo"""
    return str(Path(__file__).parents[2].absolute())


def get_bind_paths():
    paths = [str(path.absolute()) for path in compute_environment.PATHS.__dict__.values()]
    return paths


def get_abs_path_from_config_path(config_path):
    if config_path[0] == "/":
        abs_path = config_path
    else:
        abs_path = str((Path(get_base_path()) / config_path).absolute())
    return abs_path
