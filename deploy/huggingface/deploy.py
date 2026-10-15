"""Assemble the Hugging Face Space and upload it.

    pip install huggingface_hub
    hf auth login                      # once, with a token that has write access
    python deploy/huggingface/deploy.py <username>/deep-age-estimator

Use --stage-only <dir> to just assemble the folder (e.g. to test it with `docker build`).
"""
import argparse
import shutil
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent

MODEL_FILES = ["age_efficientnet.keras", "age_cnn_gan.keras", "age_cnn.keras", "gan_generator.keras", "metrics.json"]
IGNORE = shutil.ignore_patterns("node_modules", "dist", "__pycache__", ".pytest_cache", ".env", "tests", "*.pyc")


def stage(dest: Path) -> None:
    missing = [f for f in MODEL_FILES if not (ROOT / "models" / f).exists()]
    if missing:
        raise SystemExit(f"Missing in models/: {', '.join(missing)} — run the notebook export cell first")

    shutil.copy(HERE / "Dockerfile", dest / "Dockerfile")
    shutil.copy(HERE / "README.md", dest / "README.md")
    (dest / "backend").mkdir()
    shutil.copy(ROOT / "backend" / "requirements.txt", dest / "backend" / "requirements.txt")
    shutil.copytree(ROOT / "backend" / "app", dest / "backend" / "app", ignore=IGNORE)
    shutil.copytree(ROOT / "frontend", dest / "frontend", ignore=IGNORE)
    (dest / "models").mkdir()
    for f in MODEL_FILES:
        shutil.copy(ROOT / "models" / f, dest / "models" / f)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("space", nargs="?", help="Space id, e.g. username/deep-age-estimator")
    parser.add_argument("--stage-only", type=Path, help="Only assemble the Space folder here")
    args = parser.parse_args()

    if args.stage_only:
        args.stage_only.mkdir(parents=True, exist_ok=True)
        stage(args.stage_only)
        print(f"Staged in {args.stage_only}")
        return
    if not args.space:
        parser.error("space id is required unless --stage-only is given")

    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(args.space, repo_type="space", space_sdk="docker", exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        stage(Path(tmp))
        api.upload_folder(
            repo_id=args.space,
            repo_type="space",
            folder_path=tmp,
            commit_message="Deploy Deep Age Estimator",
        )
    print(f"Uploaded. Build logs and app: https://huggingface.co/spaces/{args.space}")


if __name__ == "__main__":
    main()
