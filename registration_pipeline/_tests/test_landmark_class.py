"""
Tests the landmarks class serialization
"""

from pathlib import Path
from pytest import raises
import numpy as np
from typing import get_args

from registration_pipeline import landmarks

Landmarks = landmarks.Landmarks

eg_array = np.array(
    [
        [11, 22, 33],
        [12, -2, 23],
        [13, 22, 33],
        [14, -2, 33],
        [15, 22, 23],
        [16, -2, 33],
        [17, 22, 33],
    ]
).astype(np.float64)
eg_names = get_args(landmarks.LandmarkName)


def test_from_array_and_names():
    lmarks = landmarks.from_array_and_names(eg_array, eg_names, True)
    assert lmarks.loc["elipsoid-body", "x"] == 33


def test_to_array_and_names():
    lmarks = landmarks.from_array_and_names(eg_array, eg_names, True)
    array, names = landmarks.to_array_and_names(lmarks, False)
    assert np.all(array == eg_array[:, ::-1])
    assert names == eg_names


def test_to_cmtk():
    lmarks = landmarks.from_array_and_names(eg_array, eg_names, True)
    cmtk_string = landmarks.to_cmtk(lmarks)


if __name__ == "__main__":
    test_to_cmtk()
