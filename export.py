"""Standalone export script. Usage: python export.py --ckpt path/to/ckpt.pt --output out.ply"""
import argparse
from pathlib import Path
from src.exporter import export_ply

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    export_ply(args.ckpt, args.output)
