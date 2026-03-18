"""
Launch ShinkaEvolve to discover transformation scoring functions.

Usage:
    python evolve/run_evo.py
    python evolve/run_evo.py --num-generations 100 --llm claude-sonnet-4-5-20250514
    python evolve/run_evo.py --llm openrouter/qwen/qwen3-coder

Requires at least one LLM API key in a .env file or environment:
    ANTHROPIC_API_KEY=...    (for Claude models)
    OPENAI_API_KEY=...       (for GPT models)
    OPENROUTER_API_KEY=...   (for OpenRouter models)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from shinka.core import ShinkaEvolveRunner, EvolutionConfig
from shinka.database import DatabaseConfig
from shinka.launch import LocalJobConfig


TASK_DIR = Path(__file__).parent
RESULTS_DIR = Path("outputs/evolve/results")

TASK_SYS_MSG = """\
You are optimizing a scoring function for medicinal chemistry SAR analysis.
The function ranks R-group modifications (structural changes to a molecule)
by how likely they are to cause large activity changes at a biological target.

This is a data-driven generalization of the 1972 Topliss decision tree.

Features (numpy array, shape (N, 8), columns):
  0: delta_MW          - molecular weight change (signed, Daltons)
  1: delta_LogP        - lipophilicity change (signed)
  2: delta_TPSA        - polar surface area change (signed, Angstrom^2)
  3: delta_HBDonors    - H-bond donor count change (signed)
  4: delta_HBAcceptors - H-bond acceptor count change (signed)
  5: delta_RotBonds    - rotatable bond count change (signed)
  6: delta_HeavyAtomCount - heavy atom count change (signed)
  7: fp_tanimoto       - fingerprint similarity between R-groups (0-1)

Scientific principles to consider:
- Large lipophilicity changes (delta_LogP) often indicate hydrophobic pocket effects
- H-bond donor/acceptor changes affect binding site interactions
- Low fp_tanimoto means structurally very different R-groups → bigger change
- Nonlinear relationships and feature interactions are likely important
- Both magnitude AND direction (sign) of changes can matter
- The goal: predict which modifications cause the LARGEST |delta_pActivity|

Constraints:
- Must use numpy vectorized operations (no Python loops over rows)
- Input: np.ndarray shape (N, 8), Output: np.ndarray shape (N,)
- Higher scores should predict more informative (cliff-causing) modifications
"""


def main():
    parser = argparse.ArgumentParser(description="Run ShinkaEvolve for transformation scoring")
    parser.add_argument(
        "--num-generations", type=int, default=50,
        help="Number of evolution generations (default: 50)",
    )
    parser.add_argument(
        "--llm", type=str, nargs="+",
        default=["claude-sonnet-4-5-20250514"],
        help="LLM model(s) for code evolution (default: claude-sonnet-4-5-20250514)",
    )
    parser.add_argument(
        "--num-islands", type=int, default=2,
        help="Number of evolutionary islands (default: 2)",
    )
    parser.add_argument(
        "--archive-size", type=int, default=30,
        help="Population archive size per island (default: 30)",
    )
    parser.add_argument(
        "--results-dir", type=Path, default=RESULTS_DIR,
        help=f"Results directory (default: {RESULTS_DIR})",
    )
    parser.add_argument(
        "--resume", type=Path, default=None,
        help="Resume from a previous results directory",
    )
    args = parser.parse_args()

    args.results_dir.mkdir(parents=True, exist_ok=True)

    # Verify eval data exists
    eval_data = TASK_DIR / "eval_data" / "eval_data.npz"
    if not eval_data.exists():
        print(f"ERROR: Evaluation data not found at {eval_data}")
        print("Run first:  python scripts/prepare_evolve_data.py")
        sys.exit(1)

    evo_config = EvolutionConfig(
        num_generations=args.num_generations,
        llm_models=args.llm,
        init_program_path=str(TASK_DIR / "initial.py"),
        language="python",
        task_sys_msg=TASK_SYS_MSG,
        results_dir=str(args.results_dir),
        meta_rec_interval=10,
    )

    db_config = DatabaseConfig(
        archive_size=args.archive_size,
        num_islands=args.num_islands,
        num_archive_inspirations=1,
        migration_interval=10,
    )

    job_config = LocalJobConfig(
        eval_program_path=str(TASK_DIR / "evaluate.py"),
        conda_env="activity-cliffs",
    )

    runner = ShinkaEvolveRunner(
        evo_config=evo_config,
        job_config=job_config,
        db_config=db_config,
        max_evaluation_jobs=1,
    )

    if args.resume:
        print(f"Resuming from {args.resume} ...")
        runner.load_from_results_dir(str(args.resume))

    print(f"Starting ShinkaEvolve: {args.num_generations} generations, "
          f"LLMs: {args.llm}, islands: {args.num_islands}")
    print(f"Results → {args.results_dir}")
    runner.run()


if __name__ == "__main__":
    main()
