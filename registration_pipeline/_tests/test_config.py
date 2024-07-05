"""
tests the config class
"""

from pathlib import Path
import tempfile

import pytest
import numpy as np

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)

from registration_pipeline.registration_config import (
    RegistrationConfig,
    find_cmtk,
    TemplateImageInfo,
    get_appropriate_image_path,
)
import registration_pipeline._tests as tests

tests.test_plugin.make_test_template(DATA_DIR / "test_template")

config = RegistrationConfig(
    template_path=DATA_DIR / "test_template",
    cmtk_exe_dir=find_cmtk(),
    out_dir=Path("out0"),
    ncpu=8,
)


def test_get_landmarks():
    landmarks = config.get_landmarks()
    assert isinstance(landmarks.index[0], np.str_)


def test_find_cmtk():
    out = find_cmtk()
    assert out is not None


def test_info_from_path(dummy_templates):
    path = Path("3892_328932_328934_none.nrrd")
    tii = TemplateImageInfo.from_path(path)


@pytest.fixture
def dummy_templates():
    paths = [
        Path("100_100_200_left.nhdr"),
        Path("380_380_380_left.nhdr"),
        Path("519_519_1000_right.nhdr"),
        Path("519_519_1000_both.nhdr"),
        Path("100_100_200_right.nhdr"),
        Path("380_380_380_none.nhdr"),
        Path("100_100_200_none.nhdr"),
        Path("100_100_200_right.raw.gz"),
        Path("380_380_380_both.raw.gz"),
        Path("519_519_1000_right.raw.gz"),
        Path("519_519_1000_none.nhdr"),
        Path("100_100_200_none.raw.gz"),
        Path("380_380_380_right.nhdr"),
        Path("380_380_380_right.raw.gz"),
        Path("519_519_1000_none.raw.gz"),
        Path("380_380_380_left.raw.gz"),
        Path("519_519_1000_both.raw.gz"),
        Path("100_100_200_left.raw.gz"),
        Path("original-path"),
        Path("519_519_1000_left.nhdr"),
        Path("100_100_200_both.raw.gz"),
        Path("519_519_1000_left.raw.gz"),
        Path("100_100_200_both.nhdr"),
        Path("380_380_380_none.raw.gz"),
        Path("380_380_380_both.nhdr"),
    ]
    out = Path(tempfile.gettempdir())
    for path in paths:
        (out / path).touch()
    return out


def test_gaip_first(dummy_templates):
    moving_tii = TemplateImageInfo(0.6, 0.6, 1.0, "left")
    out_path = get_appropriate_image_path(dummy_templates, moving_tii)
    assert out_path.name == "519_519_1000_left.nhdr"


def test_gaip_middle(dummy_templates):
    moving_tii = TemplateImageInfo(0.38, 0.38, 0.5, "none")
    out_path = get_appropriate_image_path(dummy_templates, moving_tii)
    assert out_path.name == "380_380_380_none.nhdr"


def test_gaip_end(dummy_templates):
    moving_tii = TemplateImageInfo(0.01, 5.0, 0.5, "both")
    out_path = get_appropriate_image_path(dummy_templates, moving_tii)
    assert out_path.name == "100_100_200_both.nhdr"


if __name__ == "__main__":
    test_gaip_end(dummy_templates())
