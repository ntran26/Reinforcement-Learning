import gymnasium
import matplotlib.pyplot as plt
import numpy as np

import flappy_bird_gymnasium
from flappy_bird_gymnasium.envs.utils import MODEL_PATH
from flappy_bird_gymnasium.tests.dueling import DuelingDQN
from flappy_bird_gymnasium.tests.dueling_v2 import DuelingDQN as DuelingDQN_v2
from flappy_bird_gymnasium.tests.framestack import FrameStack

plt.ion()


def play(
    epoch=500, audio_on=True, render_mode="human", use_lidar=True, score_limit=None
):
    env = gymnasium.make(
        "FlappyBird-v0",
        audio_on=audio_on,
        render_mode=render_mode,
        use_lidar=use_lidar,
        score_limit=score_limit,
    )

    # init models
    if use_lidar:
        env = FrameStack(env, 16)
        q_model = DuelingDQN_v2(env.action_space.n, 2, 128, 4, 6)
        q_model.build((None, *env.observation_space.shape))
        q_model.load_weights(MODEL_PATH + "/LIDAR_AVG_16steps_15px.h5")
    else:
        q_model = DuelingDQN(env.action_space.n)
        q_model.build((None, *env.observation_space.shape))
        q_model.load_weights(MODEL_PATH + "/model.h5")

    q_model.summary()

    # if render_mode == "human" and use_lidar:
    #     similarity_scores = np.dot(
    #         q_model.pos_embs.position[0], np.transpose(q_model.pos_embs.position[0])
    #     ) / (
    #         np.linalg.norm(q_model.pos_embs.position[0], axis=-1)
    #         * np.linalg.norm(q_model.pos_embs.position[0], axis=-1)
    #     )
    #     similarity_scores[similarity_scores < 0.5] = 0.5
    #
    #     plt.figure(dpi=300)
    #     plt.imshow(similarity_scores, cmap="inferno", interpolation="nearest")
    #     plt.title("Positional Embedding")
    #     plt.ylabel("Timestep")
    #     plt.xlabel("Timestep")
    #     plt.colorbar()
    #     plt.pause(10)

    # run
    for t in range(epoch):
        state, _ = env.reset()
        state = np.expand_dims(state, axis=0)
        while True:
            # Getting action
            if use_lidar:
                action, attn_matrix = q_model.get_action(state)
            else:
                action = q_model.get_action(state)
            action = np.array(action, copy=False, dtype=env.env.action_space.dtype)

            # if render_mode == "human" and use_lidar:
            #     # plotting the attention matrix
            #     plt.imshow(attn_matrix[0, 0], cmap="inferno", interpolation="nearest")
            #     plt.title("Attention head 0")
            #     plt.ylabel("Timestep")
            #     plt.xlabel("Timestep")
            #     plt.pause(0.001)

            # Processing action
            next_state, _, done, truncated, info = env.step(action)

            state = np.expand_dims(next_state, axis=0)
            # print(f"Obs: {state}\n" f"Action: {action}\n" f"Score: {info['score']}\n")

            if done or truncated:
                break

        print(f"Epoch: {t}, Score: {info['score']}")

    env.close()
    assert state.shape == (1,) + env.observation_space.shape
    assert info["score"] > 0


def test_play():
    play(epoch=1, audio_on=False, render_mode=None, use_lidar=False, score_limit=10)
    play(epoch=1, audio_on=False, render_mode=None, use_lidar=True, score_limit=10)


if __name__ == "__main__":
    play()
