"""
Contains the class for storing config
"""

from __future__ import annotations
from pathlib import Path
import platform
import os

import nrrd
from attrs import frozen, field
from attrs.validators import instance_of
import numpy as np

from registration_pipeline import landmarks


def find_cmtk() -> Path | None:
    """
    returns the cmtk executable or None if it cannot be found
    """
    search_path = [Path(p) for p in os.environ["PATH"].split(os.pathsep) if len(p) > 0]
    search_path += [
        Path().home() / "bin",
        Path().home() / "Downloads/usr/lib/cmtk/bin",
        Path("/usr/lib/cmtk/bin/"),
        Path("/usr/local/lib/cmtk/bin"),
        Path("/usr/local/bin"),
        Path("/opt/local/bin"),
        Path("/opt/local/lib/cmtk/bin/"),
        Path("/Applications/IGSRegistrationTools/bin"),
    ]
    if platform.system() == "Windows":
        search_path += [
            Path(r"C:\cygwin64\usr\local\lib\cmtk\bin"),
            Path(r"C:\Program Files\CMTK-3.3\CMTK\lib\cmtk\bin"),
        ]
    for path in search_path:
        if not path.is_dir():
            continue

        try:
            return next(path.glob("reformatx*")).resolve().parent
        except StopIteration:
            continue
    # failed to find a cmtk directory
    return None


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
    cmtk_exe_dir: Path = field(validator=instance_of(Path))
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

    def get_cmtk_exe(self, exe: str):
        if os.name == "nt":
            return self.cmtk_exe_dir / f"{exe}.exe"
        return self.cmtk_exe_dir / exe
