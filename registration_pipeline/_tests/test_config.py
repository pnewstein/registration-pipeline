"""
tests the config class
"""

from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)

from registration_pipeline.registration_config import RegistrationConfig, find_cmtk
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


if __name__ == "__main__":
    test_get_landmarks()
