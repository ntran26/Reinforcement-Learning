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

""" Tests the simple-observations version of the Flappy Bird environment with a
human player.
"""

import gymnasium
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import pygame

import flappy_bird_gymnasium


def play(use_lidar=True):
    env = gymnasium.make(
        "FlappyBird-v0", audio_on=True, render_mode="human", use_lidar=use_lidar
    )

    steps = 0
    video_buffer = []

    obs = env.reset()
    while True:
        # Getting action:
        action = 0
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
            if event.type == pygame.KEYDOWN and (
                event.key == pygame.K_SPACE or event.key == pygame.K_UP
            ):
                action = 1

        # Processing:
        obs, _, done, _, info = env.step(action)
        video_buffer.append(obs)

        steps += 1
        print(
            f"Obs: {obs}\n"
            f"Action: {action}\n"
            f"Score: {info['score']}\n Steps: {steps}\n"
        )

        if done:
            break

    env.close()

    if use_lidar:
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, projection="polar")
        x = np.linspace((np.pi / 2), -(np.pi / 2), 180)
        y = np.array(video_buffer)
        (line,) = ax.plot(x, y[0], "-")
        ax.set_ylim([0, 1])
        ax.set_title("LIDAR scan", fontdict={"fontweight": "bold"})

        def animate(i):
            # RADAR
            line.set_ydata(y[i])
            return (line,)

        anim = animation.FuncAnimation(
            fig, animate, repeat=True, frames=steps, interval=150
        )
        anim.save("video.gif")


if __name__ == "__main__":
    play()
