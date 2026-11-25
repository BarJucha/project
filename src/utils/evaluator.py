"""Benchmarking evaluator for decentralized graph navigation agents."""
from typing import List, Tuple, Dict, Any, Callable
import tqdm

from src.env.graph_env import GraphNavEnv
from src.utils.metrics import analyze_trajectory, summarize_evaluation_results


def evaluate_agent(
    agent,
    env: GraphNavEnv,
    evaluation_pairs: List[Tuple[int, int]],
    show_progress: bool = False,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Evaluates an agent on a fixed set of source-target evaluation pairs.
    
    Args:
        agent: Agent object with select_action(obs) method or callable policy.
        env: GraphNavEnv simulation instance.
        evaluation_pairs: List of (source, target) tuples to evaluate.
        show_progress: Whether to display a tqdm progress bar.
        
    Returns:
        summary: Aggregated metrics dictionary (R_trunc, R_oracle, CI, etc.).
        episode_records: Full list of per-episode detailed records.
    """
    records = []
    iterator = tqdm.tqdm(evaluation_pairs, desc=f"Evaluating {getattr(agent, 'name', 'Agent')}") if show_progress else evaluation_pairs
    
    for src, tgt in iterator:
        # Reset agent state if agent maintains internal memory (e.g. TabuWalker, GRU, DeepSets)
        if hasattr(agent, "reset"):
            agent.reset(src, tgt)
            
        obs, info = env.reset(source=src, target=tgt)
        terminated = False
        truncated = False
        
        while not (terminated or truncated):
            if hasattr(agent, "select_action"):
                action = agent.select_action(obs)
            else:
                action = agent(obs)
                
            obs, reward, terminated, truncated, info = env.step(action)
            
        traj_diag = analyze_trajectory(info["trajectory"])
        rec = {
            "source": src,
            "target": tgt,
            "steps": info["steps"],
            "shortest_path_len": info["shortest_path_len"],
            "terminated": info["terminated"],
            "truncated": info["truncated"],
            "is_success": info["is_success"],
            "trajectory": info["trajectory"],
            "revisit_rate": traj_diag["revisit_rate"],
            "ping_pong_count": traj_diag["ping_pong_count"],
            "cycle_count": traj_diag["cycle_count"],
        }
        records.append(rec)
        
    summary = summarize_evaluation_results(records)
    summary["agent_name"] = getattr(agent, "name", str(agent))
    return summary, records
