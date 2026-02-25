#!/usr/bin/env python3
"""
Download training ABC files from matt20013/tunes.
Run from project root: python scripts/download_training_data.py

If the repo is private, copy maplewood.abc and maplewood_other.abc from
https://github.com/matt20013/tunes/tree/master/abcs into data/ manually.
"""
import os
import sys
import urllib.request

REPO_RAW = "https://raw.githubusercontent.com/matt20013/tunes/master/abcs"
FILES = ["maplewood.abc", "maplewood_other.abc"]
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    for name in FILES:
        url = f"{REPO_RAW}/{name}"
        path = os.path.join(DATA_DIR, name)
        try:
            urllib.request.urlretrieve(url, path)
            print(f"Downloaded {path}")
        except Exception as e:
            print(f"Failed to download {url}: {e}", file=sys.stderr)
            print("You can copy the file from the GitHub repo into data/ manually.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
