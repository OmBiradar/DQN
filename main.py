#!/usr/bin/env python3
import os
import typing as tt

import gymnasium as gym            # Modern RL envs (new step/reset API) [see adapter below]
import gym as gym_legacy           # Legacy Gym base class for PTAN type-check
import ale_py
import ptan

import torch
import torch.optim as optim
from ignite.engine import Engine

from lib import dqn_model, common
from lib import ignite_rl as ptan_ignite
from lib.wrappers import FrameStack   # Local, version-agnostic frame stack (CHW)

# Try Gymnasium's AtariPreprocessing if available
try:
    from gymnasium.wrappers.atari_preprocessing import AtariPreprocessing
    HAVE_ATARI_PRE = True
except Exception:
    HAVE_ATARI_PRE = False

NAME = "01_baseline"

BEST_PONG = common.Hyperparams(
    env_name="ale_py:ALE/Pong-v5",
    stop_reward=18.0,
    run_name="pong",
    replay_size=100_000,
    replay_initial=10_000,
    target_net_sync=1000,
    epsilon_frames=100_000,
    epsilon_final=0.02,
    learning_rate=9.932831968547505e-05,
    gamma=0.98,
    episodes_to_solve=340,
    checkpoint_freq=20,
    save_checkpoints=True,
)

# Adapter: expose legacy Gym API and type for PTAN (step -> 4-tuple, reset -> obs only)
class GymV21Adapter(gym_legacy.Env):
    def __init__(self, env: gym.Env):
        super().__init__()
        self.env = env
        # Reuse action/observation spaces as-is (Gymnasium spaces work for shape/n)
        self.action_space = env.action_space
        self.observation_space = env.observation_space

    def step(self, action):
        out = self.env.step(action)
        # Convert (obs, reward, terminated, truncated, info) -> (obs, reward, done, info)
        if isinstance(out, tuple) and len(out) == 5:
            obs, reward, terminated, truncated, info = out
            done = bool(terminated) or bool(truncated)
            return obs, reward, done, info
        return out

    def reset(self, **kwargs):
        out = self.env.reset(**kwargs)
        # Convert (obs, info) -> obs
        if isinstance(out, tuple) and len(out) == 2:
            return out[0]
        return out

    # Delegate optional methods
    def render(self, *args, **kwargs):
        return self.env.render(*args, **kwargs)

    def close(self):
        return self.env.close()

def train(params: common.Hyperparams,
          device: torch.device, _: dict) -> tt.Optional[int]:
    # 1) Base env with frameskip disabled (AtariPreprocessing will apply frameskip=4)
    gym.register_envs(ale_py)
    env = gym.make(params.env_name, render_mode="human", frameskip=1)  # avoid double frameskip [web:133][web:156]

    # 2) Gymnasium-native Atari preprocessing (grayscale, 84x84, frameskip=4, life-loss terminals)
    if HAVE_ATARI_PRE:
        env = AtariPreprocessing(
            env,
            noop_max=30,
            frame_skip=4,            # apply frameskip here
            screen_size=84,
            grayscale_obs=True,
            terminal_on_life_loss=True,
            scale_obs=False,         # keep uint8 [0,255]
        )  # uses Gymnasium step/reset API [web:133][web:74]

    # 3) Stack 4 frames as channels-first (4, 84, 84) using local wrapper
    env = FrameStack(env, 4)  # CHW uint8; Gymnasium semantics internally

    # 4) Present legacy Gym API and type to PTAN (fixes tuple arity + isinstance check)
    env = GymV21Adapter(env)  # PTAN now sees legacy (obs, reward, done, info) and old Gym type [web:73][web:74]

    # Model + agent
    net = dqn_model.DQN(env.observation_space.shape, env.action_space.n).to(device)
    tgt_net = ptan.agent.TargetNet(net)
    selector = ptan.actions.EpsilonGreedyActionSelector(epsilon=params.epsilon_start)
    epsilon_tracker = common.EpsilonTracker(selector, params)
    agent = ptan.agent.DQNAgent(net, selector, device=device)

    # Experience and replay
    exp_source = ptan.experience.ExperienceSourceFirstLast(env, agent, gamma=params.gamma)
    buffer = ptan.experience.ExperienceReplayBuffer(exp_source, buffer_size=params.replay_size)
    optimizer = optim.Adam(net.parameters(), lr=params.learning_rate)

    def process_batch(engine, batch):
        optimizer.zero_grad()
        loss_v = common.calc_loss_dqn(batch, net, tgt_net.target_model, gamma=params.gamma, device=device)
        loss_v.backward()
        optimizer.step()
        epsilon_tracker.frame(engine.state.iteration)
        if engine.state.iteration % params.target_net_sync == 0:
            tgt_net.sync()
        return {"loss": loss_v.item(), "epsilon": selector.epsilon}

    engine = Engine(process_batch)
    engine.state.net = net

    # RL events, logging, and solve criteria
    common.setup_ignite(engine, params, exp_source, NAME)

    # Checkpoint on episode boundaries
    @engine.on(ptan_ignite.EpisodeEvents.EPISODE_COMPLETED)
    def save_checkpoint(trainer: Engine):
        if params.save_checkpoints and trainer.state.episode % params.checkpoint_freq == 0:
            checkpoint_path = f"models/checkpoints/{params.run_name}_{NAME}_ep{trainer.state.episode}.pth"
            os.makedirs("models/checkpoints", exist_ok=True)
            torch.save(trainer.state.net.state_dict(), checkpoint_path)
            print(f"Checkpoint saved at episode {trainer.state.episode}: {checkpoint_path}")

    # Train
    r = engine.run(common.batch_generator(buffer, params.replay_initial, params.batch_size))
    if r.solved:
        model_path = f"models/{params.run_name}_{NAME}.pth"
        os.makedirs("models", exist_ok=True)
        torch.save(net.state_dict(), model_path)
        print(f"Model saved to {model_path}")
        return r.episode

if __name__ == "__main__":
    args = common.argparser().parse_args()
    common.train_or_tune(args, train, BEST_PONG)
