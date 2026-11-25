"""Recurrent Advantage Actor-Critic (R-A2C) trainer with episodic BPTT."""
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import torch as T
import torch.nn as nn

from src.env.graph_env import GraphNavEnv
from src.utils.evaluator import evaluate_agent


class RecurrentA2CTrainer:
    """Trainer for memory-augmented (GRU / Deep Sets) and memory-less CTDE A2C agents."""

    def __init__(
        self,
        agent,
        env: GraphNavEnv,
        train_targets: List[int],
        val_pairs: List[Tuple[int, int]],
        checkpoint_dir: str = "checkpoints",
        eval_every: int = 500,
        early_stopping_patience: int = 20,
        step_penalty: float = 0.0,
        grad_clip: Optional[float] = 1.0,
    ):
        self.agent = agent
        self.env = env
        self.train_targets = train_targets
        self.val_pairs = val_pairs
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.eval_every = eval_every
        self.early_stopping_patience = early_stopping_patience
        self.step_penalty = step_penalty
        self.grad_clip = grad_clip

        self.best_oracle_ratio = float("inf")
        self.patience_counter = 0

    def train_episode(self) -> Dict[str, float]:
        """Runs one episode with online state-value evaluation and full-trajectory BPTT."""
        if hasattr(self.agent, "train"):
            self.agent.train()
        else:
            self.agent.train_mode = True
            self.agent.greedy_mode = False

        obs, info = self.env.reset()
        if hasattr(self.agent, "reset"):
            self.agent.reset(obs.current_node, obs.target_node)

        tgt = info["target"]
        log_probs: List[T.Tensor] = []
        entropies: List[T.Tensor] = []
        values: List[T.Tensor] = []
        rewards: List[float] = []

        terminated = False
        truncated = False

        while not (terminated or truncated):
            probs, log_prob_dist = self.agent.get_action_distribution(obs)
            values.append(self.agent.get_value(obs.current_node, tgt))

            action_idx = int(T.multinomial(probs, 1).item())
            chosen_action = obs.neighbors[action_idx]

            log_prob = log_prob_dist[action_idx]
            entropy = -(probs * log_prob_dist).sum()

            log_probs.append(log_prob)
            entropies.append(entropy)

            obs, reward, terminated, truncated, info = self.env.step(chosen_action)
            if not terminated and self.step_penalty != 0.0:
                reward = reward + self.step_penalty
            rewards.append(reward)

            if hasattr(self.agent, "update_memory"):
                self.agent.update_memory(obs.current_node)

        num_steps = len(rewards)
        if num_steps == 0:
            return {"loss": 0.0, "actor_loss": 0.0, "critic_loss": 0.0, "steps": 0, "success": 0.0}

        if terminated:
            bootstrap_val = T.tensor(0.0, device=self.agent.device)
        else:
            with T.no_grad():
                bootstrap_val = self.agent.get_value(obs.current_node, tgt).squeeze(0)

        current_vals = T.cat(values)  # Shape (T,)
        returns_list: List[T.Tensor] = []
        for t in range(num_steps):
            next_val = bootstrap_val if t == num_steps - 1 else current_vals[t + 1].detach()
            returns_list.append(rewards[t] + self.agent.gamma * next_val)
        returns = T.stack(returns_list)

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
        if self.grad_clip is not None and self.grad_clip > 0:
            if hasattr(self.agent, "parameters_list"):
                params = self.agent.parameters_list()
            else:
                params = [p for group in self.agent.optimizer.param_groups for p in group["params"]]
            if params:
                nn.utils.clip_grad_norm_(params, max_norm=self.grad_clip)
        self.agent.optimizer.step()

        return {
            "loss": float(total_loss.item()),
            "actor_loss": float(actor_loss.item()),
            "critic_loss": float(critic_loss.item()),
            "steps": num_steps,
            "success": 1.0 if terminated else 0.0,
        }

    def train(
        self,
        num_episodes: int = 10000,
        val_subset_size: int = 100,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """Trains the agent with periodic validation, checkpointing, and early stopping."""
        history: Dict[str, List[float]] = {
            "loss": [],
            "val_oracle_ratio": [],
            "val_trunc": [],
            "val_revisit_rate": [],
            "val_ping_pong": [],
        }
        val_eval_pairs = self.val_pairs[:val_subset_size]
        t0 = time.time()

        if verbose:
            print(
                f"[RecurrentA2CTrainer] Starting training for {num_episodes} episodes on "
                f"{getattr(self.agent, 'name', 'Agent')}..."
            )

        for ep in range(1, num_episodes + 1):
            ep_metrics = self.train_episode()
            history["loss"].append(ep_metrics["loss"])

            if ep % self.eval_every == 0 or ep == num_episodes:
                if hasattr(self.agent, "eval"):
                    self.agent.eval()
                val_summary, _ = evaluate_agent(self.agent, self.env, val_eval_pairs, show_progress=False)
                r_oracle = val_summary["mean_oracle_ratio"]
                r_trunc = val_summary["r_trunc"]

                history["val_oracle_ratio"].append(r_oracle)
                history["val_trunc"].append(r_trunc)
                history["val_revisit_rate"].append(val_summary["mean_revisit_rate"])
                history["val_ping_pong"].append(val_summary["mean_ping_pong_count"])

                elapsed = time.time() - t0
                if verbose:
                    print(
                        f"  Ep {ep:>6}/{num_episodes} | Loss: {ep_metrics['loss']:.4f} | "
                        f"Val R_oracle: {r_oracle:.2f} | Val R_trunc: {r_trunc:.1f}% | "
                        f"Revisit: {val_summary['mean_revisit_rate']:.2f} | Time: {elapsed:.1f}s"
                    )

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
                            print(
                                f"[RecurrentA2CTrainer] Early stopping triggered at episode {ep} "
                                f"(patience={self.early_stopping_patience})."
                            )
                        break

        return history
