"""
tests the config class
"""

from pathlib import Path

import numpy as np

from registration_pipeline.registration_config import (
    RegistrationConfig,
    _get_landmarks_path,
)

config = RegistrationConfig(
    template_path=Path().home() / "templates/JRC2018_UNISEX",
    cmtk_exe=Path("/opt/local/bin/cmtk"),
    out_dir=Path("out0"),
    ncpu=8,
)


def test_get_landmarks():
    landmarks = config.get_landmarks()
    assert isinstance(landmarks.index[0], np.str_)


if __name__ == "__main__":
    test_get_landmarks()
