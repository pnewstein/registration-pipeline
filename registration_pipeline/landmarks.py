"""
Defines the landmarks type and its associated functions
"""

from typing import Literal, get_args, Any, Annotated, Iterable
from pathlib import Path

from attrs import frozen, field, validators
import static_frame as sf
from static_frame import CallGuard
import numpy as np
from numpy.typing import NDArray

LandmarkName = Literal[
    "left-optic-lobe",
    "right-optic-lobe",
    "right-protocerebral-bridge",
    "left-protocerebral-bridge",
    "protocerebral-bridge",
    "elipsoid-body",
    "between-antenal-lobes",
    # "left-lateralmost-cns",
    # "right-lateralmost-cns",
]


@frozen
class LandmarkInfo:
    """
    a landmark and its associated strings
    """

    landmark_names: tuple[LandmarkName, ...]
    default_layer_name: str
    button_label: str


landmark_infos = (
    LandmarkInfo(
        ("left-optic-lobe", "right-optic-lobe"),
        "optic lobes",
        "Optic Lobes (if present)",
    ),
    LandmarkInfo(
        ("left-protocerebral-bridge", "right-protocerebral-bridge"),
        "protocerebral bridge tips",
        "Protocerebral bridge tips (both required)",
    ),
    LandmarkInfo(("elipsoid-body",), "elipsoid body", "Elipsoid body"),
    LandmarkInfo(
        ("between-antenal-lobes",), "between anntenal lobes", "Between anntenal lobes"
    ),
    # LandmarkInfo(
    # ("protocerebral-bridge",), "protocerebral bridge", "Protocerebral bridge"
    # ),
    # LandmarkInfo(
    # ("left-lateralmost-cns", "right-lateralmost-cns"),
    # "lateral cns",
    # "Lateral edges of the CNS",
    # ),
)


def landmarks_validator(instance, attribute, value):
    _ = instance


def _is_landmark_name(instance, attribute, value):
    """
    asserts that field is a valid number
    """
    _ = instance
    if not value in get_args(LandmarkName):
        raise ValueError(f"{attribute.name} must be one of {get_args(LandmarkName)}")


# if TYPE_CHECKING:
# Landmarks = sf.Frame
# else:
Landmarks = sf.Frame[
    Annotated[sf.Index[Any], sf.Require.LabelsMatch(set(get_args(LandmarkName)))],
    Annotated[sf.Index[Any], sf.Require.LabelsOrder("x", "y", "z")],
    np.float64,
    np.float64,
    np.float64,
]
"""
the landmarks type columns is x, y, z
"""


def from_array_and_names(
    array: NDArray[np.float64], names: Iterable[LandmarkName], zyx: bool
) -> Landmarks:
    """
    creates a Landmarks from an array and names
    """
    if array.shape[1] != 3:
        raise TypeError("array must have 3 columns")
    if zyx:
        array = array[:, ::-1]
    return sf.Frame(array, index=names, columns=["x", "y", "z"])


def to_array_and_names(
    landmarks: Landmarks, zyx: bool
) -> tuple[NDArray[np.float64], tuple[LandmarkName]]:
    """
    converts landmars to array and names in the correct order
    """
    columns_order = ["z", "y", "x"] if zyx else ["x", "y", "z"]
    return (np.array(landmarks.loc[:, columns_order]), tuple(landmarks.index))


def to_cmtk(landmarks: Landmarks) -> str:
    """
    writes sorted landmarks in a way
    """
    sorted_landmarks = landmarks.sort_index()
    lines = [
        f"{x} {y} {z} {name}"
        for name, (x, y, z) in sorted_landmarks.iter_array_items(axis=1)
    ]
    return "\n".join(lines)
