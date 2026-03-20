"""
setup.py for task_tracker — a lightweight CLI task manager built with Click and Rich.

Install in editable mode for development:
    pip install -e tools/task_tracker

Or install directly:
    pip install tools/task_tracker
"""

from setuptools import setup, find_packages

setup(
    name="task-tracker",
    version="0.1.0",
    description="A lightweight CLI task tracker built with Click and Rich",
    author="SASVA AI",
    python_requires=">=3.10",
    # Discover all packages inside tools/task_tracker/
    packages=find_packages(exclude=["tests", "tests.*"]),
    install_requires=[
        "click>=8.1,<9.0",
        "rich>=13.0,<14.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0,<9.0",
        ],
    },
    entry_points={
        "console_scripts": [
            # Exposes the `task` command on the PATH after installation
            "task=cli:cli",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Intended Audience :: Developers",
        "Topic :: Utilities",
    ],
)
