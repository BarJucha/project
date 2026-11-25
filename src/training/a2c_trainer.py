"""Advantage Actor-Critic (A2C) trainer under Centralized Training with Decentralized Execution."""
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import torch as T

from src.env.graph_env import GraphNavEnv
from src.utils.evaluator import evaluate_agent


class A2CTrainer:
    """Trainer implementing CTDE A2C (Algorithm 1, Pisacane et al., LoG 2024)."""
    def __init__(
        self,
        agent,
        env: GraphNavEnv,
        train_targets: List[int],
        val_pairs: List[Tuple[int, int]],
        checkpoint_dir: str = "checkpoints",
        eval_every: int = 500,
        early_stopping_patience: int = 20,
    ):
        self.agent = agent
        self.env = env
        self.train_targets = train_targets
        self.val_pairs = val_pairs
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.eval_every = eval_every
        self.early_stopping_patience = early_stopping_patience
        
        self.best_oracle_ratio = float("inf")
        self.patience_counter = 0

    def train_episode(self) -> Dict[str, float]:
        """Runs a single episodic CTDE update."""
        self.agent.train_mode = True
        self.agent.greedy_mode = False
        
        # Sample fresh training pair (u_src, u_tgt in V_train)
        obs, info = self.env.reset()  # env target_nodes are set to train_targets
        
        log_probs = []
        entropies = []
        visited_states = [obs.current_node]
        rewards = []
        
        terminated = False
        truncated = False
        
        while not (terminated or truncated):
            probs, log_prob_dist = self.agent.get_action_distribution(obs)
            
            # Action sampling
            action_idx = int(T.multinomial(probs, 1).item())
            chosen_action = obs.neighbors[action_idx]
            
            # Record log prob and entropy
            log_prob = log_prob_dist[action_idx]
            entropy = -(probs * log_prob_dist).sum()
            
            log_probs.append(log_prob)
            entropies.append(entropy)
            
            obs, reward, terminated, truncated, info = self.env.step(chosen_action)
            visited_states.append(obs.current_node)
            rewards.append(reward)
            
        T_steps = len(rewards)
        if T_steps == 0:
            return {"loss": 0.0, "steps": 0, "success": 0.0}
            
        # Compute state values along trajectory
        tgt = info["target"]
        values = []
        for s in visited_states:
            val = self.agent.get_value(s, tgt)
            values.append(val)
        values = T.cat(values)  # Shape (T+1,)
        
        # Calculate 1-step TD advantages: A_hat_t = r_t + gamma * V(s_{t+1}) - V(s_t)
        # Final state V(s_T) is 0 if terminated with target reached
        returns = []
        for t in range(T_steps):
            if t == T_steps - 1 and terminated:
                next_val = T.tensor(0.0, device=self.agent.device)
            else:
                next_val = values[t + 1].detach()
            returns.append(rewards[t] + self.agent.gamma * next_val)
        returns = T.stack(returns)
        
        current_vals = values[:T_steps]
        advantages = returns - current_vals
        
        critic_loss = advantages.pow(2).mean()
        
        stacked_log_probs = T.stack(log_probs)
        stacked_entropies = T.stack(entropies)
        
        actor_loss = -(stacked_log_probs * advantages.detach()).mean()
        if self.agent.entropy_coef > 0:
            actor_loss -= self.agent.entropy_coef * stacked_entropies.mean()
            
        total_loss = actor_loss + critic_loss
        
        self.agent.optimizer.zero_grad()
        total_loss.backward()
        self.agent.optimizer.step()
        
        return {
            "loss": float(total_loss.item()),
            "actor_loss": float(actor_loss.item()),
            "critic_loss": float(critic_loss.item()),
            "steps": T_steps,
            "success": 1.0 if terminated else 0.0,
        }

    def train(
        self,
        num_episodes: int = 10000,
        val_subset_size: int = 100,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """Trains the agent for num_episodes with periodic validation and checkpointing."""
        history = {
            "loss": [],
            "val_oracle_ratio": [],
            "val_trunc": [],
        }
        val_eval_pairs = self.val_pairs[:val_subset_size]
        t0 = time.time()
        
        if verbose:
            print(f"[A2CTrainer] Starting training for {num_episodes} episodes on {getattr(self.agent, 'name', 'Agent')}...")
            
        for ep in range(1, num_episodes + 1):
            ep_metrics = self.train_episode()
            history["loss"].append(ep_metrics["loss"])
            
            if ep % self.eval_every == 0 or ep == num_episodes:
                self.agent.eval()
                val_summary, _ = evaluate_agent(self.agent, self.env, val_eval_pairs, show_progress=False)
                r_oracle = val_summary["mean_oracle_ratio"]
                r_trunc = val_summary["r_trunc"]
                
                history["val_oracle_ratio"].append(r_oracle)
                history["val_trunc"].append(r_trunc)
                
                elapsed = time.time() - t0
                if verbose:
                    print(f"  Ep {ep:>6}/{num_episodes} | Loss: {ep_metrics['loss']:.4f} | Val R_oracle: {r_oracle:.2f} | Val R_trunc: {r_trunc:.1f}% | Time: {elapsed:.1f}s")
                    
                # Save best checkpoint
                if r_oracle < self.best_oracle_ratio:
                    self.best_oracle_ratio = r_oracle
                    self.patience_counter = 0
                    best_ckpt_path = self.checkpoint_dir / f"{self.agent.name}_best.pt"
                    self.agent.save_checkpoint(str(best_ckpt_path))
                    if verbose:
                        print(f"    --> New best model saved to {best_ckpt_path} (R_oracle = {r_oracle:.2f})")
                else:
                    self.patience_counter += 1
                    if self.patience_counter >= self.early_stopping_patience:
                        if verbose:
                            print(f"[A2CTrainer] Early stopping triggered at episode {ep} (patience={self.early_stopping_patience}).")
                        break
                        
        return history
