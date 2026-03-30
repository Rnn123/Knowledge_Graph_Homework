from __future__ import annotations

import argparse
import sys
from pathlib import Path

from huggingface_hub import snapshot_download


DEFAULT_REPO_ID = "dslim/bert-base-NER"
DEFAULT_ENDPOINT = "https://hf-mirror.com"
DEFAULT_PATTERNS = [
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.txt",
    "special_tokens_map.json",
    "model.safetensors",
    "pytorch_model.bin",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download a Hugging Face NER model from a domestic mirror."
    )
    parser.add_argument(
        "--repo-id",
        default=DEFAULT_REPO_ID,
        help=f"Model repository id. Default: {DEFAULT_REPO_ID}",
    )
    parser.add_argument(
        "--endpoint",
        default=DEFAULT_ENDPOINT,
        help=f"Mirror endpoint. Default: {DEFAULT_ENDPOINT}",
    )
    parser.add_argument(
        "--output-dir",
        default=str(Path("models") / "bert-base-ner"),
        help="Local folder used to store the downloaded model files.",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="Optional Hugging Face access token. Not required for public models.",
    )
    parser.add_argument(
        "--all-files",
        action="store_true",
        help="Download all files in the repository instead of the minimal runtime set.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if the files already exist locally.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    allow_patterns = None if args.all_files else DEFAULT_PATTERNS

    print(f"Downloading model: {args.repo_id}")
    print(f"Mirror endpoint: {args.endpoint}")
    print(f"Output directory: {output_dir}")
    if allow_patterns is None:
        print("File selection: all repository files")
    else:
        print("File selection:")
        for pattern in allow_patterns:
            print(f"  - {pattern}")

    try:
        downloaded_path = snapshot_download(
            repo_id=args.repo_id,
            endpoint=args.endpoint,
            local_dir=output_dir,
            local_dir_use_symlinks=False,
            allow_patterns=allow_patterns,
            token=args.token,
            force_download=args.force,
            resume_download=True,
            max_workers=4,
        )
    except Exception as exc:
        print("\nDownload failed.")
        print(f"Reason: {exc}")
        print("\nYou can try again with:")
        print(
            "python src/download_ner_model.py "
            f'--repo-id "{args.repo_id}" --endpoint "{args.endpoint}"'
        )
        return 1

    print("\nDownload completed.")
    print(f"Saved to: {downloaded_path}")
    print("\nYou can load the model later with a local path, for example:")
    print(
        "pipeline('ner', model=r'"
        f"{output_dir}"
        "', tokenizer=r'"
        f"{output_dir}"
        "', aggregation_strategy='simple')"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
