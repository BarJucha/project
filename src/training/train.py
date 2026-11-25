"""CLI script to train GARDEN and MLP-A2C models on processed graphs."""
import argparse
import pickle
from pathlib import Path
import numpy as np

from src.env.graph_env import GraphNavEnv
from src.data.splits import load_splits_and_pairs
from src.models.garden import GARDENAgent, MLPA2CAgent
from src.training.a2c_trainer import A2CTrainer


def main():
    parser = argparse.ArgumentParser(description="Train CTDE A2C Graph Navigation Agent")
    parser.add_argument("--graph", type=str, default="facebook_1", help="Graph name in data/processed/")
    parser.add_argument("--model", type=str, default="garden", choices=["garden", "mlp", "mlp_wd"], help="Model type")
    parser.add_argument("--episodes", type=int, default=5000, help="Number of training episodes")
    parser.add_argument("--eval_every", type=int, default=500, help="Evaluation interval")
    parser.add_argument("--hidden_dim", type=int, default=64, help="Hidden dimension for Actor/Critic/GAT")
    parser.add_argument("--rep_dim", type=int, default=32, help="Representation dimension for GAT")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--entropy_coef", type=float, default=1e-3, help="Entropy coefficient lambda")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu, cuda:0)")
    args = parser.parse_args()

    graph_dir = Path("data/processed") / args.graph
    if not graph_dir.exists():
        raise FileNotFoundError(f"Processed graph directory not found: {graph_dir}")

    with open(graph_dir / "graph.pkl", "rb") as f:
        g = pickle.load(f)
    features = np.load(graph_dir / "features.npy")
    splits, val_pairs, _ = load_splits_and_pairs(str(graph_dir))

    # Initialize environment with target nodes restricted to training split
    env = GraphNavEnv(g, features, max_steps=100, target_nodes=splits["train"])

    print(f"\n[Train] Initializing {args.model.upper()} on {args.graph} ({g.number_of_nodes()} nodes)...")
    if args.model == "garden":
        agent = GARDENAgent(
            env=env,
            hidden_dim=args.hidden_dim,
            rep_dim=args.rep_dim,
            lr=args.lr,
            entropy_coef=args.entropy_coef,
            use_degree=True,
            device=args.device,
            name=f"GARDEN_{args.graph}",
        )
    elif args.model == "mlp_wd":
        agent = MLPA2CAgent(
            env=env,
            hidden_dim=args.hidden_dim,
            lr=args.lr,
            entropy_coef=args.entropy_coef,
            use_degree=True,
            device=args.device,
            name=f"MLPA2CWD_{args.graph}",
        )
    else:
        agent = MLPA2CAgent(
            env=env,
            hidden_dim=args.hidden_dim,
            lr=args.lr,
            entropy_coef=args.entropy_coef,
            use_degree=False,
            device=args.device,
            name=f"MLPA2C_{args.graph}",
        )

    trainer = A2CTrainer(
        agent=agent,
        env=env,
        train_targets=splits["train"],
        val_pairs=val_pairs,
        checkpoint_dir=f"checkpoints/{args.graph}",
        eval_every=args.eval_every,
    )

    trainer.train(num_episodes=args.episodes, val_subset_size=100, verbose=True)


if __name__ == "__main__":
    main()
