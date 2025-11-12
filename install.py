import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


def run(cmd, *, cwd=None, check=True):
    """Run a shell command and stream output."""
    print(f"[CMD] {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=cwd, check=check)
    return result.returncode


def ensure_git_repo(path: Path):
    if path.exists():
        return
    run(["git", "clone", "https://github.com/NVlabs/nvdiffrast", str(path)])


def main():
    parser = argparse.ArgumentParser(description="Install the SuGaR environment.")
    parser.add_argument(
        "--env-name",
        default="sugar",
        help="Name of the conda environment to create or update (default: sugar).",
    )
    parser.add_argument(
        "--skip-nvdiffrast",
        action="store_true",
        help="Skip cloning and installing the nvdiffrast dependency.",
    )
    parser.add_argument(
        "--force-recreate",
        action="store_true",
        help="Remove any existing environment with the same name before creating a new one.",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent
    env_file = project_root / "environment.yml"

    if not env_file.exists():
        print(f"[ERROR] environment.yml not found at {env_file}")
        return 1

    if not shutil.which("conda"):
        print("[ERROR] Conda is required but was not found in PATH.")
        return 1

    if args.force_recreate:
        run(["conda", "env", "remove", "--name", args.env_name, "--yes"], check=False)

    print(f"[INFO] Creating/updating conda environment '{args.env_name}'...")
    run(
        [
            "conda",
            "env",
            "create",
            "--name",
            args.env_name,
            "--file",
            str(env_file),
            "--force",
        ]
    )

    print("[INFO] Upgrading pip...")
    run(
        ["conda", "run", "-n", args.env_name, "python", "-m", "pip", "install", "--upgrade", "pip"]
    )

    diff_gaussian = project_root / "gaussian_splatting" / "submodules" / "diff-gaussian-rasterization"
    simple_knn = project_root / "gaussian_splatting" / "submodules" / "simple-knn"

    print("[INFO] Installing diff-gaussian-rasterization...")
    run(
        [
            "conda",
            "run",
            "-n",
            args.env_name,
            "python",
            "-m",
            "pip",
            "install",
            "--no-deps",
            "--no-build-isolation",
            "--editable",
            str(diff_gaussian),
        ]
    )

    print("[INFO] Installing simple-knn...")
    run(
        [
            "conda",
            "run",
            "-n",
            args.env_name,
            "python",
            "-m",
            "pip",
            "install",
            "--no-deps",
            "--no-build-isolation",
            "--editable",
            str(simple_knn),
        ]
    )

    if not args.skip_nvdiffrast:
        print("[INFO] Installing nvdiffrast (can be skipped with --skip-nvdiffrast)...")
        nvdiffrast_path = project_root / "nvdiffrast"
        ensure_git_repo(nvdiffrast_path)
        run(
            [
                "conda",
                "run",
                "-n",
                args.env_name,
                "python",
                "-m",
                "pip",
                "install",
            "--no-build-isolation",
                "--editable",
                str(nvdiffrast_path),
            ]
        )
    else:
        print("[INFO] Skipping nvdiffrast installation.")

    print("[INFO] Setting up environment variables for CUDA and PyTorch...")
    # Get the conda environment path
    try:
        result = subprocess.run(
            ["conda", "run", "-n", args.env_name, "python", "-c", "import sys; print(sys.prefix)"],
            capture_output=True,
            text=True,
            check=True,
        )
        env_path = result.stdout.strip()
    except subprocess.CalledProcessError:
        # Fallback: try to get from conda info
        result = subprocess.run(
            ["conda", "info", "--envs"],
            capture_output=True,
            text=True,
            check=True,
        )
        for line in result.stdout.splitlines():
            if line.startswith(args.env_name):
                env_path = line.split()[-1]
                break
        else:
            env_path = None

    if env_path:
        activate_dir = Path(env_path) / "etc" / "conda" / "activate.d"
        activate_dir.mkdir(parents=True, exist_ok=True)
        
        activation_script = activate_dir / "sugar_env.sh"
        activation_script.write_text(f"""#!/bin/bash
# Automatically set CUDA and PyTorch library paths when conda environment is activated

export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CUDA_HOME/bin:$PATH"

# Add PyTorch libraries to LD_LIBRARY_PATH (required for CUDA extensions)
PYTORCH_LIB_PATH="$CONDA_PREFIX/lib/python3.10/site-packages/torch/lib"
if [ -d "$PYTORCH_LIB_PATH" ]; then
    export LD_LIBRARY_PATH="$PYTORCH_LIB_PATH:$CUDA_HOME/lib:$CUDA_HOME/lib64:$LD_LIBRARY_PATH"
else
    export LD_LIBRARY_PATH="$CUDA_HOME/lib:$CUDA_HOME/lib64:$LD_LIBRARY_PATH"
fi

# Add SuGaR CUDA extension modules to PYTHONPATH (required for simple_knn and diff_gaussian_rasterization)
SUGAR_ROOT="{project_root}"
if [ -d "${{SUGAR_ROOT}}/gaussian_splatting/submodules/simple-knn" ]; then
    export PYTHONPATH="${{SUGAR_ROOT}}/gaussian_splatting/submodules/simple-knn:${{SUGAR_ROOT}}/gaussian_splatting/submodules/diff-gaussian-rasterization:$PYTHONPATH"
fi
""")
        activation_script.chmod(0o755)
        print(f"[INFO] Created activation script at {activation_script}")
    else:
        print("[WARNING] Could not determine environment path. Environment variables must be set manually.")

    print(
        f"[INFO] SuGaR environment '{args.env_name}' ready!\n"
        f"The environment variables (CUDA_HOME, PATH, LD_LIBRARY_PATH) will be automatically set when you run:\n"
        f"  conda activate {args.env_name}\n"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
