import json
import os

from heal_swin.utils import get_paths
from compute_environment.compute_environment import LOGGING


def get_tracking_uri():
    backend = LOGGING.mlflow_backend
    if backend == "filesystem":
        return "file://" + get_paths.get_mlruns_path()
    elif backend == "sqlite":
        server_file = get_paths.get_tracking_server_file_path()
        assert os.path.isfile(server_file), "Tracking server file not found, is the server running?"
        with open(server_file, "r") as f:
            server_data = json.load(f)
        return f"http://{server_data['host']}:{server_data['port']}"
    else:
        assert False, "Unknown mlflow backend {backend} specified in compute environment"
