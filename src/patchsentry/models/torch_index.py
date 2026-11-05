"""Exact nearest-neighbour search in PyTorch, used by the training pipeline.

Why FAISS is not used during training: on macOS, FAISS and PyTorch each load their own OpenMP
runtime, and a process holding both segfaults. This was reproduced on faiss-cpu 1.8 to 1.15, with
either import order and with ``KMP_DUPLICATE_LIB_OK``. So the two are never loaded together:

- Training process: PyTorch and this index, which runs on the GPU when there is one.
- Serving process: ONNX Runtime and FAISS, with no PyTorch.

The memory bank is stored as a plain array, so either side can build its index from it, and
``pipeline/verify_serving.py`` checks that both give the same metrics.
"""

from __future__ import annotations

import numpy as np
import torch


def default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


class TorchExactIndex:
    """Exact L2 search with the same ``search`` contract as a FAISS index."""

    def __init__(self, bank: np.ndarray, device: str | None = None, chunk_size: int = 8192) -> None:
        self.device = device or default_device()
        self.chunk_size = chunk_size
        self.bank = torch.from_numpy(np.ascontiguousarray(bank, dtype=np.float32)).to(self.device)
        self.bank_squared = (self.bank**2).sum(dim=1)
        self.ntotal = bank.shape[0]

    @torch.inference_mode()
    def search(self, queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Return (squared distances, indices), both of shape (len(queries), k)."""
        distances, indices = [], []
        for start in range(0, len(queries), self.chunk_size):
            chunk = torch.from_numpy(
                np.ascontiguousarray(queries[start : start + self.chunk_size], dtype=np.float32)
            ).to(self.device)
            # ||x - b||^2 = ||x||^2 + ||b||^2 - 2 x.b, which is one matmul instead of a double loop.
            squared = (
                (chunk**2).sum(dim=1, keepdim=True)
                + self.bank_squared[None, :]
                - 2.0 * chunk @ self.bank.T
            ).clamp_min(0.0)
            values, positions = torch.topk(squared, k, dim=1, largest=False)
            distances.append(values.cpu().numpy())
            indices.append(positions.cpu().numpy())
        return np.concatenate(distances), np.concatenate(indices)
