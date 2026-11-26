"""Safely normalize Fish Speech's FIREFLY_GAN_VQ_CONFIG without importing Python model code."""

import argparse
import ast
import json
from pathlib import Path


def normalize_config(source: str) -> dict:
    if len(source.encode("utf-8")) > 1024 * 1024:
        raise ValueError("Firefly source config exceeds 1 MiB")
    tree = ast.parse(source)
    matches = []
    for statement in tree.body:
        if isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Constant) and isinstance(statement.value.value, str):
            continue
        if not isinstance(statement, ast.Assign) or len(statement.targets) != 1 or not isinstance(statement.targets[0], ast.Name) or statement.targets[0].id != "FIREFLY_GAN_VQ_CONFIG":
            raise ValueError("source must contain only a static Firefly configuration assignment")
        matches.append(ast.literal_eval(statement.value))
    if len(matches) != 1 or not isinstance(matches[0], dict):
        raise ValueError("expected exactly one literal FIREFLY_GAN_VQ_CONFIG mapping")
    value = matches[0]
    if not all(isinstance(key, str) for key in value):
        raise ValueError("invalid Firefly config keys")
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    config = normalize_config(args.source.read_text(encoding="utf-8"))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(config, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
