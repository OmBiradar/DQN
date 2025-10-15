#!/usr/bin/env python3
# filepath: /home/ink/Desktop/DQN/gui.py
import os
import gymnasium as gym
import ptan
import torch
import time
import argparse
import tkinter as tk
from tkinter import ttk, filedialog

from lib import dqn_model, common
from main import BEST_PONG, NAME

def get_available_models():
    """Get list of available model files"""
    models = []
    
    # Check for final solved model
    final_model = f"models/{BEST_PONG.run_name}_{NAME}.pth"
    if os.path.exists(final_model):
        models.append(("Final model", final_model))
    
    # Check for checkpoint models
    checkpoint_dir = "models/checkpoints"
    if os.path.exists(checkpoint_dir):
        for file in sorted(os.listdir(checkpoint_dir)):
            if file.endswith(".pth"):
                episode = file.split("_ep")[-1].split(".")[0]
                models.append((f"Checkpoint Episode {episode}", os.path.join(checkpoint_dir, file)))
    
    return models

def play_model(model_path, delay=0.02, episodes=5):
    """Play the game with the selected model"""
    print(f"Loading model from {model_path}")
    device = torch.device("cpu")  # CPU is fine for inference
    
    # Create environment with human rendering
    env = gym.make(BEST_PONG.env_name, render_mode="human")
    env = ptan.common.wrappers.wrap_dqn(env)
    
    # Load model
    net = dqn_model.DQN(env.observation_space.shape, env.action_space.n).to(device)
    net.load_state_dict(torch.load(model_path, map_location=device))
    print(f"Model loaded successfully")
    
    # Create agent with zero exploration (pure exploitation)
    selector = ptan.actions.ArgmaxActionSelector()  # No exploration
    agent = ptan.agent.DQNAgent(net, selector, device=device)
    
    # Play several episodes
    for episode in range(episodes):
        obs, _ = env.reset()
        total_reward = 0.0
        done = False
        steps = 0
        
        print(f"Starting episode {episode+1}/{episodes}")
        
        while not done:
            # Slow down rendering to make it watchable
            time.sleep(delay)
            
            # Get action from our agent
            actions, _ = agent([obs])
            action = actions[0]
            
            # Execute action in environment
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
            steps += 1
        
        print(f"Episode {episode+1}: reward={total_reward}, steps={steps}")

class ModelSelector:
    def __init__(self, root):
        self.root = root
        self.root.title("DQN Pong Visualizer")
        self.root.geometry("600x500")
        
        # Create the main frame
        main_frame = ttk.Frame(self.root, padding=20)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Title
        ttk.Label(
            main_frame, 
            text="DQN Pong Agent Visualizer", 
            font=("Arial", 16, "bold")
        ).pack(pady=(0, 20))
        
        # Get available models
        self.models = get_available_models()
        
        if not self.models:
            ttk.Label(
                main_frame,
                text="No trained models found.\nPlease train models first by running main.py",
                font=("Arial", 12),
                justify=tk.CENTER
            ).pack(pady=20)
            
            # Refresh button
            ttk.Button(
                main_frame,
                text="Refresh Model List",
                command=self.refresh_models
            ).pack(pady=10)
        else:
            self.create_model_selection(main_frame)
    
    def refresh_models(self):
        """Refresh the list of available models"""
        for widget in self.root.winfo_children():
            widget.destroy()
        self.__init__(self.root)
        
    def create_model_selection(self, parent_frame):
        """Create the model selection interface"""
        # Create model selection frame
        model_frame = ttk.LabelFrame(parent_frame, text="Select Trained Model", padding=10)
        model_frame.pack(fill=tk.X, pady=10)
        
        # Add model options
        self.model_var = tk.StringVar()
        for i, (label, path) in enumerate(self.models):
            ttk.Radiobutton(
                model_frame,
                text=f"{label}",
                value=path,
                variable=self.model_var
            ).pack(anchor=tk.W, pady=2)
            
            # Show file path in smaller text
            path_label = ttk.Label(
                model_frame, 
                text=f"    Path: {path}", 
                font=("Arial", 8),
                foreground="gray"
            )
            path_label.pack(anchor=tk.W, pady=(0, 5))
        
        # Default select first model
        if self.models:
            self.model_var.set(self.models[0][1])
        
        # Settings frame
        settings_frame = ttk.LabelFrame(parent_frame, text="Playback Settings", padding=10)
        settings_frame.pack(fill=tk.X, pady=10)
        
        # Episodes setting
        episodes_frame = ttk.Frame(settings_frame)
        episodes_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(episodes_frame, text="Number of Episodes:").pack(side=tk.LEFT)
        
        self.episodes_var = tk.IntVar(value=3)
        episodes_spin = ttk.Spinbox(
            episodes_frame,
            from_=1,
            to=20,
            textvariable=self.episodes_var,
            width=5
        )
        episodes_spin.pack(side=tk.LEFT, padx=10)
        
        # Delay setting
        delay_frame = ttk.Frame(settings_frame)
        delay_frame.pack(fill=tk.X, pady=10)
        
        ttk.Label(delay_frame, text="Animation Speed:").pack(side=tk.LEFT)
        
        self.speed_var = tk.DoubleVar(value=0.02)
        
        ttk.Label(delay_frame, text="Slow").pack(side=tk.LEFT, padx=(10, 5))
        speed_scale = ttk.Scale(
            delay_frame,
            from_=0.1,  # Slower
            to=0.001,   # Faster
            variable=self.speed_var,
            orient=tk.HORIZONTAL
        )
        speed_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        ttk.Label(delay_frame, text="Fast").pack(side=tk.LEFT, padx=(5, 0))
        
        # Action buttons
        button_frame = ttk.Frame(parent_frame)
        button_frame.pack(fill=tk.X, pady=(20, 0))
        
        ttk.Button(
            button_frame,
            text="Browse for Model...",
            command=self.load_custom_model
        ).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(
            button_frame,
            text="Refresh",
            command=self.refresh_models
        ).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(
            button_frame,
            text="Play Model",
            command=self.play_selected_model,
            style="Accent.TButton"
        ).pack(side=tk.RIGHT, padx=5)
        
        # Instructions
        instruction_text = (
            "Instructions:\n"
            "1. Select a model from the list or browse for a custom model\n"
            "2. Adjust playback settings if needed\n"
            "3. Click 'Play Model' to watch the agent play\n"
            "4. The game will open in a separate window"
        )
        
        instruction_frame = ttk.LabelFrame(parent_frame, text="Help", padding=10)
        instruction_frame.pack(fill=tk.X, pady=(20, 0))
        
        ttk.Label(
            instruction_frame,
            text=instruction_text,
            justify=tk.LEFT,
            wraplength=550
        ).pack(fill=tk.X)
    
    def play_selected_model(self):
        """Play the selected model"""
        selected_model = self.model_var.get()
        if not selected_model:
            return
        
        try:
            # Disable the window while playing
            self.root.title("DQN Pong Visualizer - Playing...")
            
            # Run the game visualization
            play_model(
                model_path=selected_model,
                delay=self.speed_var.get(),
                episodes=self.episodes_var.get()
            )
        except Exception as e:
            import traceback
            traceback.print_exc()
            tk.messagebox.showerror("Error", f"Error playing model: {str(e)}")
        finally:
            # Restore the window title
            self.root.title("DQN Pong Visualizer")
    
    def load_custom_model(self):
        """Open file dialog to select a custom model file"""
        file_path = filedialog.askopenfilename(
            title="Select Model File",
            filetypes=[("PyTorch Models", "*.pth"), ("All Files", "*.*")],
            initialdir="models"
        )
        
        if file_path:
            self.model_var.set(file_path)

def main():
    parser = argparse.ArgumentParser(description="DQN Pong Agent Visualizer")
    parser.add_argument("--model", help="Path to model file (skips GUI)")
    parser.add_argument("--episodes", type=int, default=3, help="Number of episodes to play")
    parser.add_argument("--delay", type=float, default=0.02, help="Delay between frames (seconds)")
    args = parser.parse_args()
    
    if args.model and os.path.exists(args.model):
        # Direct play with command line arguments
        play_model(args.model, delay=args.delay, episodes=args.episodes)
    else:
        # Show GUI selector
        root = tk.Tk()
        app = ModelSelector(root)
        root.mainloop()

if __name__ == "__main__":
    main()