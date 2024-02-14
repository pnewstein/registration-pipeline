"""
Contains the class for storing config
"""

from __future__ import annotations
from pathlib import Path
import nrrd

from attrs import frozen, field
from attrs.validators import instance_of
import numpy as np

from registration_pipeline import landmarks


def _get_landmarks_path(template_path: Path) -> Path:
    return template_path / "landmarks.csv"


def _get_original_template_path(template_path: Path) -> Path:
    return next(template_path.glob(template_path.name + ".*"))


@frozen()
class RegistrationConfig:
    "config object for the pipeline"

    template_path: Path = field(validator=instance_of(Path))
    """
    folder containing:
        orignial template image
        landmarks.json (with coords relative to orignial template json)
        sudirectory-scaled:
            several scaled images in nhdr format
    """
    cmtk_exe: Path = field(validator=instance_of(Path))
    """
    the path to the cmtk executable
    """
    out_dir: Path = field(validator=instance_of(Path))
    """
    the directory to write all files
    """
    ncpu: int = field(validator=instance_of(int))
    """
    the number of cpus to use
    """

    def get_landmarks(self) -> landmarks.Landmarks:
        """
        gets the lanmarks assocated with the config template
        """
        return landmarks.Landmarks.from_csv(
            _get_landmarks_path(self.template_path), index_depth=1
        )

    def get_original_scale_xyz(self) -> np.ndarray:
        """
        returns the scale of the template
        """
        head = nrrd.read_header(str(_get_original_template_path(self.template_path)))
        return np.diag(head["space directions"])

    def get_cmtk_transforms_path(self) -> Path:
        """
        the path where all the cmtk transforms will go
        """
        out = self.out_dir / "cmtk-transforms"
        out.mkdir(parents=True, exist_ok=True)
        return out
