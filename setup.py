#!/usr/bin/env python3
from setuptools import find_packages, setup

setup(
    name="heal_swin",
    version="0.1",
    authors=["Jan Gerken", "Oscar Carlsson", "Hampus Linander", "Heiner Spieß"],
    description="Original implementation of HEAL-SWIN: A vision transformer on the sphere",
    packages=find_packages(include=["heal_swin", "heal_swin.*"]),
    install_requires=[
        "astropy==5.1",
        "chamfer-distance @ git+https://github.com/otaheri/"
        "chamfer_distance@f86f6f7cadd3aca642704573d1626c67ca2e2846",
        "databricks-cli==0.14.3",
        "dill==0.3.4",
        "einops==0.4.0",
        "entrypoints==0.4",
        "google-auth==1.30.1",
        "google-auth-oauthlib==0.4.4",
        "healpy==1.15.2",
        "matplotlib==3.3.4",
        "mlflow==1.29.0",
        "torch==1.8.0",
        "numpy==1.19.2",
        "opencv-python-headless==4.4.0.46",
        "pandas==1.1.4",
        "packaging==20.9",
        "protobuf==3.14.0",
        "pytorch-lightning==1.3.4",
        "PyYAML==5.4.1",
        "scipy==1.6.0",
        "timm==0.4.12",
        "tensorboard==2.4.1",
        "torchmetrics==0.3.2",
        "torchvision==0.9.0",
        "tqdm==4.53.0",
        "yapf==0.32.0",
    ],
    python_requires=">=3.8, <3.9",
    setup_requires=[
        "setuptools_scm",
    ],
    extras_require={
        "test": ["pytest==6.2.2"],
        "formatting": ["black==22.10.0"],
        "dev": ["flake8==3.7.9"],
    },
)
