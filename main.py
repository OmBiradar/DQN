#!/usr/bin/env python3
import os
import gymnasium as gym # Core RL library provides environment to develop and test RL models
import ptan # PyTorch add-on which provides RL specific utilities like Replay buffers, agent wrappers etc.
import typing as tt # makes it easy to debug by providing type hinting

import torch # PyTorch
import torch.optim as optim # Contains various optimization algorithms

from ignite.engine import Engine # Works with PyTorch to simplify training and validation loops and logging metrics

from lib import dqn_model, common # Local model imports

import ptan.ignite as ptan_ignite

NAME = "01_baseline"

BEST_PONG = common.Hyperparams(
    env_name="PongNoFrameskip-v4",
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
    checkpoint_freq=20,  # Save a checkpoint every 20 episodes
    save_checkpoints=True,
)


def train(params: common.Hyperparams,
          device: torch.device, _: dict) -> tt.Optional[int]:
    env = gym.make(params.env_name, render_mode="human")
    env = ptan.common.wrappers.wrap_dqn(env)

    net = dqn_model.DQN(env.observation_space.shape, env.action_space.n).to(device)
    tgt_net = ptan.agent.TargetNet(net)
    selector = ptan.actions.EpsilonGreedyActionSelector(epsilon=params.epsilon_start)
    epsilon_tracker = common.EpsilonTracker(selector, params)
    agent = ptan.agent.DQNAgent(net, selector, device=device)

    exp_source = ptan.experience.ExperienceSourceFirstLast(
        env, agent, gamma=params.gamma, env_seed=common.SEED)
    buffer = ptan.experience.ExperienceReplayBuffer(
        exp_source, buffer_size=params.replay_size)
    optimizer = optim.Adam(net.parameters(), lr=params.learning_rate)

    def process_batch(engine, batch):
        optimizer.zero_grad()
        loss_v = common.calc_loss_dqn(batch, net, tgt_net.target_model,
                                      gamma=params.gamma, device=device)
        loss_v.backward()
        optimizer.step()
        epsilon_tracker.frame(engine.state.iteration)
        if engine.state.iteration % params.target_net_sync == 0:
            tgt_net.sync()
        return {
            "loss": loss_v.item(),
            "epsilon": selector.epsilon,
        }

    engine = Engine(process_batch)
    engine.state.net = net

    # Call setup_ignite FIRST to register the custom events
    common.setup_ignite(engine, params, exp_source, NAME)

    # THEN attach your custom event handler
    @engine.on(ptan_ignite.EpisodeEvents.EPISODE_COMPLETED)
    def save_checkpoint(trainer: Engine):
        if trainer.state.episode % params.checkpoint_freq == 0:
            checkpoint_path = f"models/checkpoints/{params.run_name}_{NAME}_ep{trainer.state.episode}.pth"
            os.makedirs("models/checkpoints", exist_ok=True)
            torch.save(trainer.state.net.state_dict(), checkpoint_path)
            print(f"Checkpoint saved at episode {trainer.state.episode}: {checkpoint_path}")

    # Run the engine
    r = engine.run(common.batch_generator(buffer, params.replay_initial, params.batch_size))
    if r.solved:
        # Save the model when solved
        model_path = f"models/{params.run_name}_{NAME}.pth"
        os.makedirs("models", exist_ok=True)
        torch.save(net.state_dict(), model_path)
        print(f"Model saved to {model_path}")
        return r.episode


if __name__ == "__main__":
    args = common.argparser().parse_args()
    common.train_or_tune(args, train, BEST_PONG)