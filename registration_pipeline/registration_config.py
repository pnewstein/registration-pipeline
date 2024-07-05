"""
Contains the class for storing config
"""

from __future__ import annotations
from pathlib import Path
from typing import Literal, get_args, cast
import platform
import os
import re

import nrrd
from attrs import frozen, field
from attrs.validators import instance_of
from attr import asdict
import numpy as np
import static_frame as sf

from registration_pipeline import landmarks

OpticLobeCondition = Literal["none", "left", "right", "both"]


def get_scale(path: Path) -> np.ndarray:
    """
    gets the scale of a nhdr file
    """
    head = nrrd.read_header(str(path))
    return np.diag(head["space directions"])


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
    op_path = template_path / "orignial-path"
    return template_path / op_path.read_text("utf-8")


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
        return get_scale(_get_original_template_path(self.template_path))

    def get_cmtk_transforms_path(self) -> Path:
        """
        the path where all the cmtk transforms will go
        """
        out = self.out_dir / "cmtk-transforms"
        out.mkdir(parents=True, exist_ok=True)
        return out

    def get_cmtk_exe(self, exe: str):
        """
        returns the path to cmtk or cmtk.exe depending on your os
        """
        if os.name == "nt":
            return self.cmtk_exe_dir / f"{exe}.exe"
        return self.cmtk_exe_dir / exe


@frozen()
class TemplateImageInfo:
    """
    describes an individual template image
    """

    x_scale: float = field(validator=instance_of(float))
    y_scale: float = field(validator=instance_of(float))
    z_scale: float = field(validator=instance_of(float))
    optic_lobe_condition: OpticLobeCondition = field()

    @classmethod
    def from_path(cls, path: Path) -> "TemplateImageInfo":
        """
        parses a path to get the information
        """
        regex_match = re.match(
            r"(\d+)_(\d+)_(\d+)_(left|right|both|none)\.(nrrd|nhdr)", path.name
        )
        if regex_match is None:
            raise ValueError(f"invalid path: {path}")
        x_str, y_str, z_str, optic_lobe_condition, _ = regex_match.groups()
        if optic_lobe_condition not in get_args(OpticLobeCondition):
            raise ValueError(f"invalid path: {path}")
        optic_lobe_condition = cast(OpticLobeCondition, optic_lobe_condition)
        return cls(
            x_scale=int(x_str) / 1000,
            y_scale=int(y_str) / 1000,
            z_scale=int(z_str) / 1000,
            optic_lobe_condition=optic_lobe_condition,
        )

    def matches_header(self, path: Path) -> bool:
        """
        checks whether the info matches the xyz of the header
        """
        xyz = get_scale(path)
        return np.array_equal(np.round(xyz, 3), np.round(self.xyz(), 3))

    def xyz(self) -> np.ndarray:
        """
        returns the xyz as an array
        """
        return np.array([self.x_scale, self.y_scale, self.z_scale])

    def to_path(self) -> Path:
        """
        the path encoding this information
        demplate path must be added later
        """
        integered_scale = np.round(self.xyz() * 1000).astype(int)
        scale_str = "_".join(str(i) for i in integered_scale)
        return Path(f"{scale_str}_{self.optic_lobe_condition}.nhdr")


def get_appropriate_image_path(
    template_path: Path, moving_tii: TemplateImageInfo, smallness_coef=1.1
) -> Path:
    """
    gets a a template whoes size in 3 dimentions times smallness_coef is larger than the
    image moving tii and has the same optic lobe condition

    returns the path to align moving image to
    """
    tiis = (TemplateImageInfo.from_path(path) for path in template_path.glob("*.nhdr"))
    records = [{k: v for k, v in asdict(tii).items()} for tii in tiis]
    raw_frame = sf.Frame.from_dict_records(records)
    ol_filtered = raw_frame.loc[
        raw_frame["optic_lobe_condition"] == moving_tii.optic_lobe_condition
    ]
    frame = (
        ol_filtered.insert_after(
            "optic_lobe_condition",
            sf.Series(
                ol_filtered["x_scale"]
                * ol_filtered["y_scale"]
                * ol_filtered["z_scale"],
                name="pixel_area",
            ),
        )
        .insert_after(
            "pixel_area",
            sf.Series(
                (ol_filtered["x_scale"] < (moving_tii.x_scale * smallness_coef))
                & (ol_filtered["y_scale"] < (moving_tii.y_scale * smallness_coef))
                & (ol_filtered["z_scale"] < (moving_tii.z_scale * smallness_coef)),
                name="small_enough",
            ),
        )
        .sort_values("pixel_area", ascending=False)
    )
    (frame_inds,) = np.where(frame["small_enough"])
    if len(frame_inds) == 0:
        record = frame.iloc[-1]
    else:
        record = frame.iloc[frame_inds[0]]
    fixed_tii = TemplateImageInfo(
        x_scale=record["x_scale"],
        y_scale=record["y_scale"],
        z_scale=record["z_scale"],
        optic_lobe_condition=record["optic_lobe_condition"],
    )
    return template_path / fixed_tii.to_path()
