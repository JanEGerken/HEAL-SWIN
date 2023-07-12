#!/usr/bin/env python3

import argparse
import subprocess
import sys
import os
import json
import getpass
import datetime
import socket
from pathlib import Path

from compute_environment import compute_environment
from heal_swin.utils import get_paths


def assert_mlflow_db_exists():
    db_path = get_paths.get_mlflow_db_path()
    os.makedirs(Path(db_path).parent, exist_ok=True)
    if not os.path.isfile(db_path):
        open(db_path, "w").close()


def env_prefix(env):
    if env == "local":
        return []
    if "singularity" in env:
        container_name = compute_environment.CONTAINER.singularity_container_name
        container_path = os.path.join(get_paths.get_container_path(), container_name)
        command = ["singularity", "exec", "--nv"]
        if env == "singularity":
            command += ["--no-home"]
        command += [
            "--env",
            "PYTHONPATH=$PYTHONPATH:" + get_paths.get_base_path(),
        ]
        command += ["--env", "GUNICORN_CMD_ARGS='--timeout 180'"]
        if "MASTER_PORT" in os.environ:
            command += ["--env", "MASTER_PORT=" + os.environ["MASTER_PORT"]]
        if "MASTER_ADDR" in os.environ:
            command += ["--env", "MASTER_ADDR=" + os.environ["MASTER_ADDR"]]
        command += ["--env", "MPLCONFIGDIR=" + get_paths.get_mpl_cache_path()]
        for path in get_paths.get_bind_paths():
            command += ["--bind", path]
        command += [container_path]
        return command
    if env == "docker":
        command = [
            "docker",
            "run",
            "-it",
            "-u",
            f"{os.getuid()}:{os.getgid()}",
            "--network",
            "host",
        ]
        command += ["--env", "PYTHONPATH=$PYTHONPATH:" + get_paths.get_s2cnn_path()]
        if "MASTER_PORT" in os.environ:
            command += ["--env", "MASTER_PORT=" + os.environ["MASTER_PORT"]]
        if "MASTER_ADDR" in os.environ:
            command += ["--env", "MASTER_ADDR=" + os.environ["MASTER_ADDR"]]
        for path in get_paths.get_bind_paths():
            command += ["--mount", "type=bind,src=" + path + ",dst=" + path]
        command += ["-w", get_paths.get_base_path()]
        command += ["heal_swin"]
        return command


def mlf_server(run_args, sub_args):
    if sub_args != []:
        print(f"unknown arguments: {' '.join(sub_args)}")
        sys.exit(1)

    if run_args.backend == "filesystem":
        command = env_prefix(run_args.env) + ["mlflow", "server"]
        command += ["--backend-store-uri", "file://" + get_paths.get_mlruns_path()]
        command += ["--workers", str(run_args.workers)]
        command += ["--port", str(run_args.port)]
        print(f"running: {' '.join(command)}")
        subprocess.run(command)
        return

    server_file = get_paths.get_tracking_server_file_path()
    if os.path.isfile(server_file):
        with open(server_file, "r") as f:
            server_data = json.load(f)
        print(
            f"The tracking server is already running on the host {server_data['host']},"
            + f" listening to port {server_data['port']}. It was started"
            + f" at {server_data['start_time']} by the user {server_data['user']}. Aborting."
        )
        sys.exit(1)

    server_data = {
        "user": getpass.getuser(),
        "start_time": datetime.datetime.now().strftime("%H:%M:%S %d-%m-%Y"),
        "host": socket.gethostname(),
        "port": run_args.port,
        "workers": run_args.workers,
        "timeout": run_args.timeout,
    }
    with open(server_file, "w") as f:
        json.dump(server_data, f)

    command = env_prefix(run_args.env) + ["mlflow", "server"]
    command += ["--backend-store-uri"]
    command += ["sqlite:///" + get_paths.get_mlflow_db_path() + "?timeout=" + str(run_args.timeout)]
    command += ["--default-artifact-root", "file://" + get_paths.get_mlruns_path()]
    command += ["--workers", str(run_args.workers)]
    command += ["--host", "0.0.0.0"]
    command += ["--port", str(run_args.port)]
    print(f"running: {' '.join(command)}")
    try:
        subprocess.run(command)
    except KeyboardInterrupt:
        pass

    if os.path.isfile(server_file):
        os.remove(server_file)
        print(f"removed server file {server_file}")


def build_sin(run_args, sub_args):
    if sub_args != []:
        print(f"unknown arguments: {' '.join(sub_args)}")
        sys.exit(1)
    containers_path = os.path.join(get_paths.get_base_path(), "containers")
    output_path = get_paths.get_container_path()
    container_name = compute_environment.CONTAINER.singularity_container_name
    output_file = os.path.join(output_path, container_name)
    command = ["singularity", "build"]
    if run_args.tmpdir is not None:
        command += ["--tmpdir", run_args.tmpdir]
    command += [output_file, "singularity_recipe"]
    print(f"running: {' '.join(command)}")
    subprocess.run(command, cwd=containers_path)


def build_docker(run_args, sub_args):
    if sub_args != []:
        print(f"unknown arguments: {' '.join(sub_args)}")
        sys.exit(1)
    container_path = get_paths.get_container_path()
    command = ["docker", "build"]
    command += ["-t", "heal_swin", "."]
    print(f"running: {' '.join(command)}")
    subprocess.run(command, cwd=container_path)


def bash(run_args, sub_args):
    command = env_prefix(run_args.env) + ["/bin/bash"]
    command += sub_args
    print(f"running: {' '.join(command)}")
    subprocess.run(command)


def test_repo(run_args, sub_args):
    command = env_prefix(run_args.env) + ["pytest", "-x"]
    command += sub_args
    print(f"running: {' '.join(command)}")
    subprocess.run(command)


def train(run_args, sub_args):
    path = Path(__file__).parent.joinpath("heal_swin/train.py").absolute()
    command = env_prefix(run_args.env) + ["python3", "-u", str(path)]
    command += sub_args
    print(f"running: {' '.join(command)}")
    subprocess.run(command)


def resume(run_args, sub_args):
    path = Path(__file__).parent.joinpath("heal_swin/resume.py").absolute()
    command = env_prefix(run_args.env) + ["python3", "-u", str(path)]
    command += sub_args
    print(f"running: {' '.join(command)}")
    subprocess.run(command)


def evaluate(run_args, sub_args):
    path = Path(__file__).parent.joinpath("heal_swin/evaluate.py").absolute()
    command = env_prefix(run_args.env) + ["python3", "-u", str(path)]
    command += sub_args
    print(f"running: {' '.join(command)}")
    subprocess.run(command)


def format_code(run_args, sub_args):
    # Run black with line-length 100
    command = env_prefix(run_args.env) + [
        "black",
        "-l",
        "100",
        get_paths.get_base_path(),
    ]
    command += sub_args
    print(f"running: {' '.join(command)}")
    subprocess.run(command)


def main():
    compute_environment.inform()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env", choices=["local", "singularity", "singularity_vscode", "docker"], default="local"
    )

    sp = parser.add_subparsers(dest="subparser_name")

    train_parser = sp.add_parser("bash")
    train_parser.set_defaults(func=bash)

    mlf_server_parser = sp.add_parser("start-mlflow-server")
    mlf_server_parser.add_argument("--port", type=int, default=5000)
    mlf_server_parser.add_argument("--workers", type=int, default=1)
    mlf_server_parser.add_argument("--timeout", type=int, default=30)
    mlf_server_parser.add_argument(
        "--backend", type=str, choices=["sqlite", "filesystem"], default="sqlite"
    )
    mlf_server_parser.set_defaults(func=mlf_server)

    build_sin_parser = sp.add_parser("build-singularity")
    build_sin_parser.add_argument("--tmpdir", type=str, default=None)
    build_sin_parser.set_defaults(func=build_sin)

    build_docker_parser = sp.add_parser("build-docker")
    build_docker_parser.set_defaults(func=build_docker)

    test_parser = sp.add_parser("test-repo")
    test_parser.set_defaults(func=test_repo)

    train_parser = sp.add_parser("train")
    train_parser.set_defaults(func=train)

    resume_parser = sp.add_parser("resume")
    resume_parser.set_defaults(func=resume)

    evaluate_parser = sp.add_parser("evaluate")
    evaluate_parser.set_defaults(func=evaluate)

    format_code_parser = sp.add_parser("format-code")
    format_code_parser.set_defaults(func=format_code)

    run_args, sub_args = parser.parse_known_args()

    assert_mlflow_db_exists()

    run_args.func(run_args, sub_args)


if __name__ == "__main__":
    main()
