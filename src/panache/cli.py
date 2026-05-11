from __future__ import annotations

import argparse

from .config import load_run_config
from .runner import run_batch


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Standalone river plume detection runner.")
    parser.add_argument("config", help="Path to a JSON config file.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_run_config(args.config)
    results_path = run_batch(config)
    print(results_path)


if __name__ == "__main__":
    main()
