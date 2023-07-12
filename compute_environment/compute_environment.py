import os

try:
    from compute_environment import current_environment as env
except (ModuleNotFoundError, ImportError):
    print("Could not import current_environment.py")
    print("Using default environment with defaults from local_environment.py")
    from compute_environment import local_environment as env


PATHS = env.PATHS
CONTAINER = env.CONTAINER
LOGGING = env.LOGGING


def inform():
    print("--------- Compute environment --------")
    print(f"Using environment specified in {env.__file__}")
    print("Project paths:")
    for name, path in PATHS.__dict__.items():
        print(f"  {name}: {path} {path.absolute()}")
    print("Container specifications:")
    for name, value in CONTAINER.__dict__.items():
        print(f"  {name}: {value}")
    print("Logging specifications:")
    for name, value in LOGGING.__dict__.items():
        print(f"  {name}: {value}")
    print(
        "**** These values can be specified by creating (or copying) a new file",
        "current_environnment.py ****",
    )

    first = True
    for name, path in PATHS.__dict__.items():
        if not os.path.isdir(path.absolute()):
            if first:
                print("-" * 80)
            first = False
            print(f"{name} directory does not exist, creating {path.absolute()}")
            os.makedirs(path.absolute())
            print("-" * 80)

    print("\n\n")
