from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from patchsentry.data.download import is_extracted
from patchsentry.data.preprocessing import IMAGENET_MEAN, load_mask, normalize_batch, resize_rgb
from patchsentry.data.visa import load_category, read_split, split_calibration


def _make_visa(root: Path, normal_train: int = 12) -> None:
    (root / "split_csv").mkdir(parents=True)
    (root / "widget").mkdir()
    lines = ["object,split,label,image,mask"]

    def save(name: str, value: int) -> str:
        Image.new("RGB", (40, 30), (value, value, value)).save(root / "widget" / name)
        return f"widget/{name}"

    for i in range(normal_train):
        lines.append(f"widget,train,normal,{save(f'train{i}.png', 100)},")
    lines.append(f"widget,test,normal,{save('good.png', 100)},")
    mask = Image.new("L", (40, 30), 0)
    mask.paste(7, (10, 10, 20, 20))  # VisA masks use small non-zero values, not 255
    mask.save(root / "widget" / "bad_mask.png")
    lines.append(f"widget,test,anomaly,{save('bad.png', 200)},widget/bad_mask.png")
    (root / "split_csv" / "1cls.csv").write_text("\n".join(lines) + "\n")


def test_load_category_shapes_labels_and_masks(tmp_path: Path) -> None:
    _make_visa(tmp_path)
    data = load_category(tmp_path, "widget", image_size=16, calibration_fraction=0.25)
    assert data.train.shape == (9, 16, 16, 3)
    assert data.calibration.shape == (3, 16, 16, 3)
    assert data.test.shape == (2, 16, 16, 3)
    assert data.test_labels.tolist() == [False, True]
    assert not data.test_masks[0].any()
    assert data.test_masks[1].any(), "non-zero mask values must count as anomalous"


def test_unknown_category_is_an_error(tmp_path: Path) -> None:
    _make_visa(tmp_path)
    with pytest.raises(ValueError, match="not found"):
        load_category(tmp_path, "nope", image_size=16)


def test_missing_split_file_explains_how_to_fix_it(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match=r"patchsentry\.data\.download"):
        read_split(tmp_path)


def test_bad_rows_are_rejected_instead_of_skipped(tmp_path: Path) -> None:
    (tmp_path / "split_csv").mkdir()
    csv_path = tmp_path / "split_csv" / "1cls.csv"
    csv_path.write_text("object,split,label,image,mask\nw,train,weird,a.png,\n")
    with pytest.raises(ValueError, match="Unknown label"):
        read_split(tmp_path)
    csv_path.write_text("object,split,label,image,mask\nw,test,anomaly,a.png,\n")
    with pytest.raises(ValueError, match="without a mask"):
        read_split(tmp_path)
    csv_path.write_text("object,split\nw,train\n")
    with pytest.raises(ValueError, match="lacks columns"):
        read_split(tmp_path)


def test_calibration_split_is_disjoint_complete_and_deterministic() -> None:
    train, calibration = split_calibration(100, 0.25, seed=3)
    assert len(calibration) == 25
    assert set(train).isdisjoint(calibration)
    assert set(train) | set(calibration) == set(range(100))
    again_train, _ = split_calibration(100, 0.25, seed=3)
    assert np.array_equal(train, again_train)
    with pytest.raises(ValueError, match="fraction"):
        split_calibration(10, 1.0, seed=0)


def test_resize_converts_any_mode_to_rgb() -> None:
    for mode in ("L", "RGBA", "P", "CMYK"):
        assert resize_rgb(Image.new(mode, (10, 20)), 8).shape == (8, 8, 3)


def test_mask_stays_binary_after_resizing(tmp_path: Path) -> None:
    mask = Image.new("L", (100, 100), 0)
    mask.paste(255, (0, 0, 50, 100))
    path = tmp_path / "mask.png"
    mask.save(path)
    assert load_mask(path, 10).dtype == bool


def test_normalize_batch_layout_and_values() -> None:
    images = np.full((2, 4, 4, 3), 255, dtype=np.uint8)
    output = normalize_batch(images)
    assert output.shape == (2, 3, 4, 4)
    assert output.dtype == np.float32
    assert output[0, 0, 0, 0] == pytest.approx((1.0 - IMAGENET_MEAN.ravel()[0]) / 0.229)
    with pytest.raises(ValueError, match="NHWC"):
        normalize_batch(images[0])


def test_incomplete_extraction_is_not_treated_as_done(tmp_path: Path) -> None:
    (tmp_path / "split_csv").mkdir()
    (tmp_path / "split_csv" / "1cls.csv").touch()
    assert not is_extracted(tmp_path)
    (tmp_path / ".complete").touch()
    assert is_extracted(tmp_path)
