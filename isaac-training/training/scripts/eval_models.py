import argparse
import json
import os
import shlex
import subprocess
import sys
import tempfile


# Edit these defaults if you want to run the script directly without CLI args.
DEFAULT_NUM_RUNS = 10
DEFAULT_SEED_START = 0
DEFAULT_OUTPUT = "/home/lzx/project/NavRL/eval_results.json"
DEFAULT_MODELS = [
    {
        "label": "baseline",
        "repo_root": "/home/lzx/project/NavRL（复件）",
        "checkpoint": "/home/lzx/project/NavRL（复件）/wandb/latest-run/run-20260407_222747-p4c0i9bu-final-saved-20260408_052908/checkpoint_final.pt",
        "overrides": [],
    },
    {
        "label": "decoder",
        "repo_root": "/home/lzx/project/NavRL",
        "checkpoint": "/home/lzx/project/NavRL/wandb/run-20260409_131728-jhne6g9f/run-20260409_131728-jhne6g9f-final-saved-20260410_020947/checkpoint_final.pt",
        "overrides": [],
    },
]


def parse_model_spec(spec: str):
    parts = spec.split("::")
    if len(parts) < 3:
        raise ValueError(
            "Model spec must be 'label::repo_root::checkpoint_path' or "
            "'label::repo_root::checkpoint_path::override1||override2'"
        )
    label, repo_root, checkpoint = parts[:3]
    overrides = []
    if len(parts) >= 4 and parts[3]:
        overrides = [item for item in parts[3].split("||") if item]
    return {
        "label": label,
        "repo_root": repo_root,
        "checkpoint": checkpoint,
        "overrides": overrides,
    }


def build_worker_command(worker: str, extra_args: list[str]):
    explicit = os.environ.get("NAVRL_EVAL_PYTHON")
    python_exe = explicit or sys.executable
    isaac_setup = "/home/lzx/isaac-sim/setup_conda_env.sh"
    if os.path.exists(isaac_setup):
        cmd = "source {} && {} {}".format(
            shlex.quote(isaac_setup),
            shlex.quote(python_exe),
            " ".join(shlex.quote(arg) for arg in [worker, *extra_args]),
        )
        return ["/bin/bash", "-lc", cmd]
    return [python_exe, worker, *extra_args]


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate multiple NavRL checkpoints over multiple seeds."
    )
    parser.add_argument(
        "--model",
        action="append",
        default=[],
        help=(
            "Model spec: "
            "'label::repo_root::checkpoint_path' or "
            "'label::repo_root::checkpoint_path::override1||override2'"
        ),
    )
    parser.add_argument("--num-runs", type=int, default=DEFAULT_NUM_RUNS)
    parser.add_argument("--seed-start", type=int, default=DEFAULT_SEED_START)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    worker = os.path.join(script_dir, "eval_checkpoint_worker.py")

    models = [parse_model_spec(spec) for spec in args.model] if args.model else list(DEFAULT_MODELS)
    if not models:
        raise ValueError(
            "No models to evaluate. Either pass --model ... on the command line "
            "or edit DEFAULT_MODELS in eval_models.py."
        )

    results = []
    for model in models:
        print(f"[Eval] evaluating {model['label']} ...")
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            tmp_path = tmp.name
        worker_args = [
            "--repo-root",
            model["repo_root"],
            "--checkpoint",
            model["checkpoint"],
            "--label",
            model["label"],
            "--num-runs",
            str(args.num_runs),
            "--seed-start",
            str(args.seed_start),
            "--output",
            tmp_path,
        ]
        for override in model["overrides"]:
            worker_args.extend(["--override", override])
        cmd = build_worker_command(worker, worker_args)
        subprocess.run(cmd, check=True)
        if not os.path.exists(tmp_path) or os.path.getsize(tmp_path) == 0:
            raise RuntimeError(
                f"Evaluation worker for {model['label']} finished without writing results to {tmp_path}."
            )
        with open(tmp_path, "r", encoding="utf-8") as f:
            results.append(json.load(f))
        os.remove(tmp_path)

    payload = {
        "num_runs": args.num_runs,
        "seed_start": args.seed_start,
        "models": results,
    }
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
