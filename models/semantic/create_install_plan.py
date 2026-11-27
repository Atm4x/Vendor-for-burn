"""Resolve an offline semantic worker install plan for a Hugging Face model."""
import argparse
import hashlib
import json
import re
from urllib.parse import quote
from urllib.request import urlopen


HUB = "https://huggingface.co"
REPOSITORY_ID = re.compile(r"[\w.-]+(?:/[\w.-]+)?")


def read(url):
    with urlopen(url, timeout=60) as response:
        return response.read()


def resolve(args):
    if not REPOSITORY_ID.fullmatch(args.repo) or any(
        part in {".", ".."} for part in args.repo.split("/")
    ):
        raise ValueError("Expected a Hugging Face model ID")
    repo = quote(args.repo)
    revision = quote(args.revision, safe="")
    metadata = json.loads(read(f"{HUB}/api/models/{repo}/revision/{revision}?blobs=true"))
    commit = metadata["sha"]
    if not re.fullmatch(r"[a-f0-9]{40}", commit):
        raise ValueError("Hub did not return a pinned commit")
    base = f"{HUB}/{repo}/resolve/{commit}"
    files = {entry["rfilename"]: entry for entry in metadata["siblings"]}
    download = []

    def artifact(role, argument, path):
        if path not in files:
            raise ValueError(f"Missing {path} in {args.repo}@{commit}")
        entry = files[path]
        url = f"{base}/{quote(path)}"
        if entry.get("lfs"):
            size = entry["size"]
            digest = entry["lfs"]["sha256"]
            content = None
        else:
            if entry["size"] > 64 * 1024 * 1024:
                raise ValueError(f"No LFS SHA-256 recorded for {path}")
            content = read(url)
            size = len(content)
            if size != entry["size"]:
                raise ValueError(f"Size changed for {path}")
            digest = hashlib.sha256(content).hexdigest()
        download.append(dict(role=role, argument=argument, path=path, url=url, size_bytes=size, sha256=digest))
        return content

    config = json.loads(artifact("config", "--config", "config.json"))
    architecture = config.get("model_type")
    if architecture not in {"qwen3", "bert", "distilbert", "roberta", "xlm-roberta"}:
        raise ValueError(f"Unsupported architecture {architecture!r}")
    if args.backend != "cpu" and architecture != "qwen3":
        raise ValueError(f"{architecture} requires --backend cpu")
    artifact("checkpoint", "--checkpoint", "model.safetensors")
    artifact("tokenizer", "--tokenizer", "tokenizer.json")
    launch = ["--model-id", args.repo, "--task", args.task, "--backend", args.backend]
    if args.task == "embedding":
        if "modules.json" in files:
            modules = json.loads(read(f"{base}/modules.json"))
            supported = {"Transformer", "Pooling", "Normalize"}
            unsupported = [module["type"] for module in modules if module["type"].rsplit(".", 1)[-1] not in supported]
            if unsupported:
                raise ValueError(f"Unsupported embedding modules: {unsupported}")
        if args.pooling:
            launch.extend(["--pooling", args.pooling])
        elif "1_Pooling/config.json" in files:
            artifact("pooling", "--pooling-config", "1_Pooling/config.json")
        elif architecture != "qwen3":
            raise ValueError("Specify --pooling for this embedding model")
        for flag, value in (("--query-prefix", args.query_prefix), ("--document-prefix", args.document_prefix)):
            if value:
                launch.extend([flag, value])
    elif args.positive_label is not None:
        launch.extend(["--positive-label", str(args.positive_label)])
    if args.max_tokens is not None:
        launch.extend(["--max-tokens", str(args.max_tokens)])
    return {
        "schema_version": 1,
        "models": [
            dict(model_id=args.repo, service=args.task, source_revision=commit, launch_arguments=launch, downloads=download)
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repo")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--task", choices=("embedding", "reranker"), required=True)
    parser.add_argument("--backend", choices=("cpu", "cuda", "wgpu"), default="cpu")
    parser.add_argument("--pooling", choices=("cls", "mean", "last"))
    parser.add_argument("--query-prefix")
    parser.add_argument("--document-prefix")
    parser.add_argument("--positive-label", type=int)
    parser.add_argument("--max-tokens", type=int, choices=range(32, 8193), metavar="32..8192")
    args = parser.parse_args()
    if args.task == "reranker" and any((args.pooling, args.query_prefix, args.document_prefix)):
        parser.error("Pooling and text prefixes apply to embeddings")
    if args.positive_label is not None and (args.positive_label < 0 or args.task != "reranker"):
        parser.error("Positive label must be a nonnegative index and applies to rerankers")
    try:
        plan = resolve(args)
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, f"{error}\n")
    print(json.dumps(plan, indent=2))


if __name__ == "__main__":
    main()
