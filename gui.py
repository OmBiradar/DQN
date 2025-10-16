#!/usr/bin/env python3
# filepath: /home/ink/Desktop/DQN/human_vs_agent.py
import os
import time
import argparse
from collections import deque
import numpy as np
import torch
import gymnasium as gym
from pettingzoo.atari import pong_v3  # two-player Pong
from pynput import keyboard  # global key listener (no root needed on X11/Wayland)

from lib import dqn_model
from main import BEST_PONG, NAME  # reuse hyperparams/run naming

# Key state
pressed = set()

def on_press(key):
    try:
        pressed.add(key.char.lower())
    except Exception:
        if key == keyboard.Key.up:
            pressed.add('up')
        elif key == keyboard.Key.down:
            pressed.add('down')
        elif key == keyboard.Key.space:
            pressed.add('space')

def on_release(key):
    try:
        pressed.discard(key.char.lower())
    except Exception:
        if key == keyboard.Key.up:
            pressed.discard('up')
        elif key == keyboard.Key.down:
            pressed.discard('down')
        elif key == keyboard.Key.space:
            pressed.discard('space')

def human_action():
    # Minimal ALE Pong action set: 0 NOOP, 1 FIRE, 2 RIGHT, 3 LEFT, 4 RIGHTFIRE, 5 LEFTFIRE
    up = 'up' in pressed
    down = 'down' in pressed
    fire = 'space' in pressed
    if up and fire:
        return 5  # LEFTFIRE maps to one direction + FIRE; exact mapping is symmetric in Pong
    if down and fire:
        return 4  # RIGHTFIRE
    if up:
        return 3  # LEFT (one vertical direction)
    if down:
        return 2  # RIGHT (the other vertical direction)
    if fire:
        return 1  # serve
    return 0  # NOOP

# Simple grayscale + resize + frame stack to match training (84x84, 4 frames, CHW)
try:
    from PIL import Image
except Exception:
    Image = None

def to_gray(img):
    # img HWC uint8
    r, g, b = img[..., 0], img[..., 1], img[..., 2]
    return (0.299 * r + 0.587 * g + 0.114 * b).astype(np.uint8)

def resize84(gray):
    if Image is None:
        # Fallback nearest-neighbor downsample (simple but OK for inference)
        h, w = gray.shape
        sh = h // 84
        sw = w // 84
        return gray[::sh, ::sw][:84, :84]
    return np.array(Image.fromarray(gray).resize((84, 84), Image.BILINEAR), dtype=np.uint8)

class ObsStacker:
    def __init__(self, num_stack=4):
        self.deque = deque(maxlen=num_stack)
        self.num_stack = num_stack

    def reset(self, first_img):
        self.deque.clear()
        gray = to_gray(first_img)
        small = resize84(gray)
        for _ in range(self.num_stack):
            self.deque.append(small)

    def push(self, img):
        gray = to_gray(img)
        small = resize84(gray)
        self.deque.append(small)

    def chw(self):
        arr = np.stack(list(self.deque), axis=0)  # (4,84,84)
        return arr

def load_model(model_path, obs_shape, n_actions, device):
    net = dqn_model.DQN(obs_shape, n_actions).to(device)
    net.load_state_dict(torch.load(model_path, map_location=device))
    net.eval()
    return net

@torch.no_grad()
def agent_action(net, obs_chw, device):
    x = torch.as_tensor(obs_chw, dtype=torch.uint8, device=device).unsqueeze(0)  # (1,4,84,84)
    x = x.float() / 255.0
    q = net(x)[0]
    return int(torch.argmax(q).item())

def play(model_path: str, episodes: int = 3, delay: float = 0.01, device_str: str = "cpu"):
    device = torch.device(device_str)
    # Two-player Pong
    env = pong_v3.env(render_mode="human", num_players=2)  # AEC two-player Pong
    # Start listener
    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()

    # Determine which agent is human vs net
    # By convention, use the second agent in env.agents as human
    # and the first as the trained net.
    net = None
    net_stacker = ObsStacker(4)
    human_stacker = ObsStacker(4)

    for ep in range(episodes):
        env.reset()
        agent_names = list(env.agents)
        if len(agent_names) < 2:
            print("Unexpected agents list, need two players; aborting.")
            break
        net_agent = agent_names[0]
        human_agent = agent_names[1]

        # Initialize stacks with first observations from both agents
        # AEC API: iterate until each agent receives its first observation
        first_obs = {}
        for agent in env.agent_iter():
            obs, reward, term, trunc, info = env.last()
            if agent not in first_obs and obs is not None:
                first_obs[agent] = obs
            # no-ops until both seen
            env.step(None if term or trunc else 0)
            if len(first_obs) == 2:
                break

        # Prepare stacks
        net_stacker.reset(first_obs[net_agent])
        human_stacker.reset(first_obs[human_agent])

        # Create/load model once shape known
        if net is None:
            # Minimal action space is 6 by default in Pong
            n_actions = 6
            net = load_model(model_path, net_stacker.chw().shape, n_actions, device)

        total = {net_agent: 0.0, human_agent: 0.0}
        steps = 0

        # Main loop
        for agent in env.agent_iter():
            obs, reward, term, trunc, info = env.last()
            if reward:
                total[agent] += float(reward)
            act = None
            if not (term or trunc):
                if agent == human_agent:
                    # Human move
                    act = human_action()
                else:
                    # Net move: update stack with latest obs then act
                    if obs is not None:
                        net_stacker.push(obs)
                    act = agent_action(net, net_stacker.chw(), device)
                steps += 1
                time.sleep(delay)
            env.step(act)

        print(f"Episode {ep+1}: net={total[net_agent]:.1f}, human={total[human_agent]:.1f}, steps={steps}")

    listener.stop()
    env.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="Path to model .pth", default=f"models/{BEST_PONG.run_name}_{NAME}.pth")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--delay", type=float, default=0.01)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"Model not found: {args.model}")
        return
    play(args.model, args.episodes, args.delay, args.device)

if __name__ == "__main__":
    main()
