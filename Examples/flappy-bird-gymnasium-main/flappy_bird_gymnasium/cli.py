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

""" Handles the initialization of the game through the command line interface.
"""

import argparse
import time

import gymnasium
import pygame

import flappy_bird_gymnasium
from flappy_bird_gymnasium.tests.test_dqn import play as dqn_agent_env
from flappy_bird_gymnasium.tests.test_human import play as human_agent_env
from flappy_bird_gymnasium.tests.test_random import play as random_agent_env


def _get_args():
    """Parses the command line arguments and returns them."""
    parser = argparse.ArgumentParser(description=__doc__)

    # Argument for the mode of execution (human or random):
    parser.add_argument(
        "--mode",
        "-m",
        type=str,
        default="human",
        choices=["human", "random", "dqn"],
        help="The execution mode for the game.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="If set, the game will be executed without rendering it.",
    )

    return parser.parse_args()


def main():
    args = _get_args()

    if args.mode == "human":
        human_agent_env()
    elif args.mode == "random":
        random_agent_env(
            audio_on=(not args.quiet), render_mode="human" if not args.quiet else None
        )
    elif args.mode == "dqn":
        dqn_agent_env(
            audio_on=(not args.quiet), render_mode="human" if not args.quiet else None
        )
    else:
        print("Invalid mode!")
