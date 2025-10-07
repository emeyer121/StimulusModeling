#!/usr/bin/env python3
"""
Setup script for stim_transformations package.
"""

from setuptools import setup, find_packages
import os

# Read the README file
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read requirements
def read_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="stim-transformations",
    version="1.0.0",
    description="A library for stimulus image transformations including centering, scaling, texture synthesis, skeletonization, and neural network feature extraction",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=read_requirements(),
)
