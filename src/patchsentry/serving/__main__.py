"""Run the service: ``python -m patchsentry.serving``."""

from __future__ import annotations

import argparse

import uvicorn

from patchsentry.serving.app import create_app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    # One worker on purpose: the micro-batcher is per process, and several workers would split the
    # traffic into smaller, less efficient batches. Scale out with replicas of the container.
    uvicorn.run(create_app(), host=args.host, port=args.port, workers=1)


if __name__ == "__main__":
    main()
