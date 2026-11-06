"""Download and extract the VisA dataset.

The archive is streamed straight into ``tarfile`` so that it never occupies disk space twice. That
matters on a laptop: the tar is 1.9 GB and would otherwise sit next to its extracted copy.

A dropped connection does not restart the download: ``_ResumingStream`` reconnects with an HTTP
Range request from the last byte read, so the tar reader never notices the interruption.
"""

from __future__ import annotations

import argparse
import logging
import tarfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

VISA_URL = "https://amazon-visual-anomaly.s3.us-west-2.amazonaws.com/VisA_20220922.tar"
SPLIT_CSV = Path("split_csv") / "1cls.csv"
_NETWORK_ERRORS = (urllib.error.URLError, TimeoutError, ConnectionError, OSError)


class _ResumingStream:
    """Minimal readable stream that survives connection drops, using HTTP Range requests."""

    def __init__(self, url: str, max_retries: int = 20, backoff_s: float = 2.0) -> None:
        self._url = url
        self._max_retries = max_retries
        self._backoff_s = backoff_s
        self._offset = 0
        self._response: Any = None

    def _connect(self) -> None:
        request = urllib.request.Request(self._url)
        if self._offset:
            request.add_header("Range", f"bytes={self._offset}-")
        self._response = urllib.request.urlopen(request, timeout=60)
        if self._offset and self._response.status != 206:
            raise RuntimeError("The server ignored the Range request; cannot resume safely.")

    def read(self, size: int = -1) -> bytes:
        for attempt in range(self._max_retries + 1):
            try:
                if self._response is None:
                    self._connect()
                data: bytes = self._response.read(size)
                self._offset += len(data)
                return data
            except _NETWORK_ERRORS as error:
                self._response = None
                if attempt == self._max_retries:
                    raise
                wait = self._backoff_s * (attempt + 1)
                logger.warning(
                    "Connection lost at byte %d (%s); retrying in %.0fs", self._offset, error, wait
                )
                time.sleep(wait)
        raise AssertionError("unreachable")


def is_extracted(root: Path) -> bool:
    """Return True if ``root`` already holds a complete extraction."""
    return (root / ".complete").exists() and (root / SPLIT_CSV).exists()


def download_visa(root: Path, url: str = VISA_URL) -> Path:
    """Stream the VisA archive from ``url`` into ``root``. Safe to call repeatedly."""
    if is_extracted(root):
        logger.info("VisA already present at %s", root)
        return root

    root.mkdir(parents=True, exist_ok=True)
    logger.info("Streaming %s into %s", url, root)
    count = 0
    # filter="data" rejects absolute paths, parent traversal, links out of the tree and device
    # files, so a tampered archive cannot write outside ``root``.
    with tarfile.open(fileobj=_ResumingStream(url), mode="r|*") as archive:  # type: ignore[call-overload]
        for member in archive:
            archive.extract(member, path=root, filter="data")
            count += 1
            if count % 2000 == 0:
                logger.info("Extracted %d members", count)

    if not (root / SPLIT_CSV).exists():
        raise RuntimeError(f"Archive did not contain {SPLIT_CSV}; refusing to mark it complete.")
    (root / ".complete").touch()
    logger.info("Done: %d members", count)
    return root


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/raw/visa"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    download_visa(args.root)


if __name__ == "__main__":
    main()
