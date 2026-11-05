"""Write result tables into README.md from the JSON reports.

Tables go between ``<!-- results:NAME -->`` and ``<!-- /results:NAME -->`` markers. No result in the
README is typed by hand, so a number there cannot drift away from the run that produced it.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

REPORTS = Path("reports")
README = Path("README.md")


def _load(name: str) -> dict[str, Any] | None:
    path = REPORTS / name
    return json.loads(path.read_text()) if path.exists() else None


def _table(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


def _number(value: float | None, digits: int = 3) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def _percent(value: float, digits: int = 1) -> str:
    return f"{100 * value:.{digits}f}%"


def sweep() -> str | None:
    report = _load("sweep.json")
    if report is None:
        return None
    results = list(report["results"])
    # Rows rerun after a bug fix replace their originals; combination rows are appended.
    rerun = _load("sweep_rerun.json")
    if rerun is not None:
        fixed = {row["variation"]: row for row in rerun["results"]}
        results = [fixed.get(row["variation"], row) for row in results]
    combinations = _load("sweep_combinations.json")
    if combinations is not None:
        results += combinations["results"]
    rows = [
        [
            row["variation"],
            _number(row["means"]["image_auroc"]),
            _number(row["means"]["aupro"]),
            _number(row["objective"]),
            _number(row["means"]["fit_seconds"], 0),
            _number(row["means"]["scoring_ms_per_image"], 1),
        ]
        for row in results
    ]
    headers = [
        "Change from baseline",
        "Image AUROC",
        "AUPRO",
        "Objective",
        "Fit (s)",
        "Search (ms/image)",
    ]
    return _table(headers, rows)


_MODEL_NAMES = {
    "patchcore": "**PatchCore (this project)**",
    "padim": "PaDiM",
    "image_knn": "Image-level k-NN",
    "autoencoder": "Convolutional autoencoder",
}


def models() -> str | None:
    report = _load("benchmark.json")
    if report is None:
        return None
    rows = []
    for family, name in _MODEL_NAMES.items():
        entry = report["summary"].get(family)
        if entry is None:
            continue
        rows.append(
            [
                name,
                _number(entry["held_out"]["image_auroc"]),
                _number(entry["held_out"]["image_f1_max"]),
                _number(entry["held_out"]["pixel_auroc"]),
                _number(entry["held_out"]["aupro"]),
                _number(entry["all"]["image_auroc"]),
                _number(entry["all"]["aupro"]),
                _number(entry["mean_fit_seconds"], 0),
            ]
        )
    headers = [
        "Model",
        "Image AUROC",
        "Image F1-max",
        "Pixel AUROC",
        "AUPRO",
        "Image AUROC (all 12)",
        "AUPRO (all 12)",
        "Fit (s)",
    ]
    return _table(headers, rows)


def per_category() -> str | None:
    report = _load("benchmark.json")
    if report is None:
        return None
    rows_by_category: dict[str, list[dict[str, Any]]] = {}
    for row in report["rows"]:
        rows_by_category.setdefault(row["category"], []).append(row)

    def mean(members: list[dict[str, Any]], prefix: str, key: str) -> float | None:
        values = [m[key] for m in members if m["model"].startswith(prefix) and m[key] is not None]
        return float(np.mean(values)) if values else None

    rows = []
    for category, members in sorted(rows_by_category.items()):
        role = "held out" if members[0]["held_out"] else "development"
        rows.append(
            [
                category,
                role,
                _number(mean(members, "patchcore", "image_auroc")),
                _number(mean(members, "padim", "image_auroc")),
                _number(mean(members, "autoencoder", "image_auroc")),
                _number(mean(members, "image_knn", "image_auroc")),
                _number(mean(members, "patchcore", "aupro")),
                _number(mean(members, "padim", "aupro")),
            ]
        )
    headers = [
        "Category",
        "Role",
        "PatchCore AUROC",
        "PaDiM AUROC",
        "Autoencoder AUROC",
        "k-NN AUROC",
        "PatchCore AUPRO",
        "PaDiM AUPRO",
    ]
    return _table(headers, rows)


def conformal() -> str | None:
    report = _load("conformal.json")
    if report is None:
        return None
    rows = [
        [
            _percent(row["alpha"]),
            _percent(row["mean_resampled_false_alarm_rate"]),
            _percent(row["mean_test_false_alarm_rate"]),
            _percent(row["mean_naive_test_false_alarm_rate"]),
            _percent(row["mean_test_recall"]),
        ]
        for row in report["summary"]
    ]
    headers = [
        "Target false alarm rate",
        "Conformal, resampled splits",
        "Conformal, official test split",
        "Naive threshold (training scores)",
        "Defects caught at this threshold",
    ]
    return _table(headers, rows)


def backbone() -> str | None:
    report = _load("bench_backbone.json")
    if report is None:
        return None
    runtimes = list(dict.fromkeys(row["runtime"] for row in report["rows"]))
    batch_sizes = sorted({row["batch_size"] for row in report["rows"]})
    lookup = {(row["runtime"], row["batch_size"]): row for row in report["rows"]}
    rows = [
        [runtime]
        + [_number(lookup[(runtime, b)]["images_per_second"], 0) for b in batch_sizes]
        + [_number(lookup[(runtime, 1)]["batch_latency_p50_ms"], 1)]
        for runtime in runtimes
    ]
    headers = ["Runtime", *[f"Images/s, batch {b}" for b in batch_sizes], "Latency, batch 1 (ms)"]
    return _table(headers, rows)


def index() -> str | None:
    report = _load("bench_index.json")
    if report is None:
        return None
    rows = []
    for kind in dict.fromkeys(row["index"] for row in report["rows"]):
        members = [row for row in report["rows"] if row["index"] == kind]

        def mean(key: str, members: list[dict[str, Any]] = members) -> float:
            return float(np.mean([m[key] for m in members]))

        rows.append(
            [
                kind,
                _number(mean("search_p50_ms_per_image"), 1),
                _number(mean("search_p95_ms_per_image"), 1),
                _number(mean("recall_at_1")),
                _number(mean("image_auroc")),
                _number(mean("build_seconds"), 1),
                _number(mean("index_size_mb"), 0),
            ]
        )
    headers = [
        "Index",
        "Search p50 (ms/image)",
        "Search p95 (ms/image)",
        "Recall@1 vs exact",
        "Image AUROC",
        "Build (s)",
        "Size (MB)",
    ]
    return _table(headers, rows)


def quantization() -> str | None:
    report = _load("serving_verification.json")
    if report is None:
        return None
    rows = []
    for model in ("float", "quantized"):
        members = [row for row in report["rows"] if row["model"] == model]
        if not members:
            continue
        rows.append(
            [
                model,
                _number(float(np.mean([m["image_auroc"] for m in members]))),
                _number(float(np.mean([m["aupro"] for m in members]))),
                _percent(float(np.mean([m["false_alarm_rate"] for m in members]))),
                _percent(float(np.mean([m["recall"] for m in members]))),
            ]
        )
    alpha = _percent(float(report["alpha"]), 0)
    headers = [
        "ONNX graph",
        "Image AUROC",
        "AUPRO",
        f"False alarm rate (target {alpha})",
        "Defects caught",
    ]
    return _table(headers, rows)


def loadtest() -> str | None:
    report = _load("loadtest.json")
    if report is None:
        return None
    rows = [
        [
            row["configuration"],
            str(row["concurrent_clients"]),
            _number(row["requests_per_second"], 1),
            _number(row["latency_p50_ms"], 0),
            _number(row["latency_p95_ms"], 0),
            _number(row["latency_p99_ms"], 0),
            _number(row["mean_batch_size"], 1),
            str(row["errors"]),
        ]
        for row in report["rows"]
    ]
    headers = [
        "Configuration",
        "Clients",
        "Requests/s",
        "p50 (ms)",
        "p95 (ms)",
        "p99 (ms)",
        "Mean batch",
        "Errors",
    ]
    return _table(headers, rows)


def drift() -> str | None:
    report = _load("drift.json")
    if report is None:
        return None
    rows = report["rows"]
    total = len(rows)

    def flagged(key: str) -> str:
        return f"{sum(row[key] < 0.001 for row in rows)} of {total}"

    table = [
        ["Clean normal images", flagged("clean_ks_p")],
        ["Normal images, 30% darker", flagged("darker_30pct_ks_p")],
        ["Normal images, defocus blur", flagged("defocus_blur_ks_p")],
    ]
    return _table(["Stream", "Categories flagged (KS p < 0.001)"], table)


SECTIONS: dict[str, Callable[[], str | None]] = {
    "sweep": sweep,
    "models": models,
    "per_category": per_category,
    "conformal": conformal,
    "backbone": backbone,
    "index": index,
    "quantization": quantization,
    "loadtest": loadtest,
    "drift": drift,
}


def main() -> None:
    text = README.read_text()
    for name, build in SECTIONS.items():
        content = build()
        if content is None:
            print(f"Skipping {name}: report not found")
            continue
        pattern = re.compile(
            rf"(<!-- results:{name} -->\n).*?(\n<!-- /results:{name} -->)", flags=re.DOTALL
        )
        replacement = r"\g<1>" + content.replace("\\", r"\\") + r"\g<2>"
        text, count = pattern.subn(replacement, text)
        if count != 1:
            raise RuntimeError(f"Expected exactly one marker pair for {name!r}, found {count}.")
    README.write_text(text)
    print("README tables updated.")


if __name__ == "__main__":
    main()
