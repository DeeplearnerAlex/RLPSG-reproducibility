"""Build the optional fixed set of native numerical kernels from source."""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import subprocess
import sys


KERNELS = {
    "accelerated_arc_batch_kernel": "accelerated_arc_batch_kernel.cpp",
    "command_projection_kernel": "command_projection.cpp",
    "warm_command_projection_kernel": "warm_command_projection.cpp",
    "native_voronoi_centroids_kernel": "native_voronoi_centroids.cpp",
    "nonideal_prediction_native_kernel_20261105": "nonideal_prediction_native.cpp",
}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compiler", default="g++", help="C++ compiler executable (default: g++)")
    parser.add_argument("--target", action="append", choices=tuple(KERNELS),
                        help="Build a named kernel; repeat to select several (default: all five)")
    parser.add_argument("--list", action="store_true", help="List source/output pairs without compiling")
    args = parser.parse_args(argv)
    source_dir = Path(__file__).resolve().parent / "src"
    suffix = ".dll" if sys.platform == "win32" else ".so"
    selected = args.target or list(KERNELS)
    if len(set(selected)) != len(selected):
        parser.error("Duplicate targets are not permitted")
    if args.list:
        for target in selected:
            print(f"{KERNELS[target]} -> {target}{suffix}")
        return 0
    compiler = shutil.which(args.compiler)
    if compiler is None:
        parser.error("C++ compiler not found; install g++ or provide --compiler")
    for target in selected:
        source = source_dir / KERNELS[target]
        output = source_dir / (target + suffix)
        if not source.is_file():
            raise FileNotFoundError(source)
        # Fixed arguments, no shell interpolation and no fast-math.  The
        # contraction flag preserves the original numerical-kernel settings.
        command = [compiler, "-std=c++17", "-O3", "-shared", "-fPIC", "-ffp-contract=off",
                   str(source), "-o", str(output)]
        subprocess.run(command, check=True)
        print(f"Built {output.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
