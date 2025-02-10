# MIT License
#
# Copyright (c) 2020 Gabriel Nogueira (Talendar)
# Copyright (c) 2023 Martin Kubovcik
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# ==============================================================================

""" Setup of the flappy-bird-gymnasium package.
"""

from typing import List

import setuptools

_VERSION = "0.4.0"

# Short description.
short_description = "A Gymnasium environment for the Flappy Bird game."

# Packages needed for the environment to run.
# The compatible release operator (`~=`) is used to match any candidate version
# that is expected to be compatible with the specified version.
REQUIRED_PACKAGES = [
    "gymnasium",
    "numpy",
    "pygame",
    "matplotlib",
]

# Packages which are only needed for testing code.
TEST_PACKAGES = ["black", "isort", "flake8", "pytest"]  # type: List[str]

# Loading the "long description" from the projects README file.
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="flappy-bird-gymnasium",
    version=_VERSION,
    author="Martin Kubovcik",
    author_email="markub3327@gmail.com",
    description=short_description,
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/markub3327/flappy-bird-gymnasium",
    project_urls={
        "Bug Tracker": "https://github.com/markub3327/flappy-bird-gymnasium/issues",
    },
    download_url="https://github.com/markub3327/flappy-bird-gymnasium/releases",
    # Contained modules and scripts:
    packages=setuptools.find_packages(),
    package_data={
        "flappy_bird_gymnasium": [
            "assets/sprites/*",
            "assets/audio/*",
            "assets/model/*",
        ]
    },
    install_requires=REQUIRED_PACKAGES,
    tests_require=REQUIRED_PACKAGES + TEST_PACKAGES,
    # PyPI package information:
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development",
        "Topic :: Software Development :: Libraries",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    license="MIT License",
    python_requires=">=3.9",
    keywords=" ".join(
        [
            "Flappy-Bird" "Game",
            "Gymnasium",
            "Farama-Foundation",
            "Reinforcement-Learning",
            "Reinforcement-Learning-Environment",
        ]
    ),
    entry_points={
        "console_scripts": [
            "flappy_bird_gymnasium = flappy_bird_gymnasium.cli:main",
        ],
    },
)
