"""PatchCore memory bank and scoring: the NumPy and FAISS half, shared with the service."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

FAISS_INDEX_KINDS = ("flat", "hnsw", "ivf")
INDEX_KINDS = (*FAISS_INDEX_KINDS, "torch")


def build_index(bank: np.ndarray, kind: str = "flat", seed: int = 0) -> Any:
    """Build a nearest-neighbour index over the bank.

    - ``flat``: exact search, cost grows linearly with bank size. The reference for accuracy.
    - ``hnsw``: navigable small-world graph, sub-linear search and more memory.
    - ``ivf``: k-means inverted lists; searches only ``nprobe`` cells.
    - ``torch``: exact search on the GPU, for the training process.

    FAISS and PyTorch are imported only when needed, because they must never share a process on
    macOS (see ``torch_index.py``).
    """
    bank = np.ascontiguousarray(bank, dtype=np.float32)
    if kind == "torch":
        from patchsentry.models.torch_index import TorchExactIndex

        return TorchExactIndex(bank)
    if kind not in FAISS_INDEX_KINDS:
        raise ValueError(f"Unknown index kind {kind!r}; expected one of {INDEX_KINDS}")

    import faiss

    dimension = bank.shape[1]
    index: Any
    if kind == "flat":
        index = faiss.IndexFlatL2(dimension)
    elif kind == "hnsw":
        index = faiss.IndexHNSWFlat(dimension, 32)
        index.hnsw.efConstruction = 200
        index.hnsw.efSearch = 64
    elif kind == "ivf":
        n_lists = max(1, int(np.sqrt(bank.shape[0])))
        index = faiss.IndexIVFFlat(faiss.IndexFlatL2(dimension), dimension, n_lists)
        index.cp.seed = seed
        index.train(bank)
        index.nprobe = max(1, n_lists // 8)
    index.add(bank)
    return index


class PatchCoreIndex:
    """Scores patch features against a memory bank.

    Image score, following Roth et al. (CVPR 2022): take the patch whose nearest bank entry is
    furthest away (``s*``), then scale it by how isolated that bank entry is:

        s = (1 - exp(d(m_test, m*)) / sum_{m in N_b(m*)} exp(d(m_test, m))) * s*

    If ``m*`` sits in a sparse part of the bank, it may be a rare normal outlier that coreset
    sampling happens to favour, so a large distance to it is weaker evidence of an anomaly.
    """

    def __init__(
        self,
        bank: np.ndarray,
        index_kind: str = "flat",
        reweight_neighbours: int = 9,
        neighbour_table: np.ndarray | None = None,
    ) -> None:
        if bank.ndim != 2 or bank.shape[0] == 0:
            raise ValueError(f"bank must be a non-empty (M, D) array, got {bank.shape}")
        self.bank = np.ascontiguousarray(bank, dtype=np.float32)
        self.index_kind = index_kind
        self.reweight_neighbours = min(reweight_neighbours, bank.shape[0])
        self.index = build_index(self.bank, index_kind)
        if neighbour_table is None:
            # Exact, and computed once at fit time so that serving never pays for it.
            exact = build_index(self.bank, "torch" if index_kind == "torch" else "flat")
            _, neighbour_table = exact.search(self.bank, self.reweight_neighbours)
        self.neighbour_table = neighbour_table

    def score(self, features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Score features (N, H, W, D): returns (image_scores (N,), patch_maps (N, H, W))."""
        if features.ndim != 4 or features.shape[-1] != self.bank.shape[1]:
            raise ValueError(
                f"Expected (N, H, W, {self.bank.shape[1]}) features, got {features.shape}"
            )
        n, height, width, dimension = features.shape
        flat = np.ascontiguousarray(features.reshape(-1, dimension), dtype=np.float32)
        squared, neighbours = self.index.search(flat, 1)
        distances = np.sqrt(np.maximum(squared[:, 0], 0.0)).reshape(n, height * width)
        neighbours = neighbours[:, 0].reshape(n, height * width)

        worst = distances.argmax(axis=1)
        rows = np.arange(n)
        s_star = distances[rows, worst]
        if self.reweight_neighbours <= 1:
            # The neighbourhood is m* alone, so the formula gives a weight of exactly 0 for every
            # image. One neighbour therefore means "no reweighting": the score is s* itself.
            return s_star.astype(np.float32), distances.reshape(n, height, width)

        patches = flat.reshape(n, height * width, dimension)[rows, worst]
        nearest_of_worst = self.neighbour_table[neighbours[rows, worst]]
        neighbour_vectors = self.bank[nearest_of_worst]  # (N, b, D)
        neighbour_distances = np.linalg.norm(neighbour_vectors - patches[:, None, :], axis=2)
        # Written in log space, because exp(d) overflows float32 at the distances seen here.
        peak = neighbour_distances.max(axis=1, keepdims=True)
        log_sum = peak[:, 0] + np.log(np.exp(neighbour_distances - peak).sum(axis=1))
        weights = 1.0 - np.exp(s_star - log_sum)
        return (weights * s_star).astype(np.float32), distances.reshape(n, height, width)

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        np.save(directory / "bank.npy", self.bank)
        np.save(directory / "neighbour_table.npy", self.neighbour_table)

    @classmethod
    def load(cls, directory: Path, index_kind: str = "flat") -> PatchCoreIndex:
        table = np.load(directory / "neighbour_table.npy", allow_pickle=False)
        return cls(
            np.load(directory / "bank.npy", allow_pickle=False),
            index_kind=index_kind,
            reweight_neighbours=table.shape[1],
            neighbour_table=table,
        )


def upsample_maps(maps: np.ndarray, image_size: int, sigma: float = 4.0) -> np.ndarray:
    """Upsample patch maps (N, H, W) to pixel resolution and smooth them with a Gaussian."""
    output = np.empty((maps.shape[0], image_size, image_size), dtype=np.float32)
    for index, patch_map in enumerate(maps):
        resized = cv2.resize(
            patch_map.astype(np.float32), (image_size, image_size), interpolation=cv2.INTER_LINEAR
        )
        output[index] = cv2.GaussianBlur(resized, (0, 0), sigmaX=sigma, sigmaY=sigma)
    return output
