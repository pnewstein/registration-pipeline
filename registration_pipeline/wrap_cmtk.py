"""
Wraps CMTK commands with python
"""

from __future__ import annotations
from subprocess import run
from pathlib import Path
from typing import Literal

from xform import CMTKtransform
import numpy as np

from registration_pipeline.registration_config import RegistrationConfig
from registration_pipeline import landmarks


RUN_KWARGS = {"check": True, "capture_output": False}


def do_landmark_registration(
    config: RegistrationConfig,
    src_landmarks: landmarks.Landmarks,
    dst_landmarks: landmarks.Landmarks,
    out: Path,
) -> float:
    """
    uses CMTK to fit a rigid registration using src_landmarks and dst_landmarks
    saves to out and also writes the landmarks to disk
    returns average distance between src and target landmark
    """
    out.parent.mkdir(parents=True, exist_ok=True)
    cmtk_command = "fit_affine_xform_landmarks"
    src_landmarks_path = out.parent / "src_landmarks.txt"
    dst_landmarks_path = out.parent / "dst_landmarks.txt"
    src_landmarks_path.write_text(landmarks.to_cmtk(src_landmarks), "utf-8")
    dst_landmarks_path.write_text(landmarks.to_cmtk(dst_landmarks), "utf-8")
    # note: we need to register dst to src because this function returns the inverse of the affine
    args = (
        config.cmtk_exe,
        cmtk_command,
        "--rigid",
        dst_landmarks_path,
        src_landmarks_path,
        out,
    )
    run(args, **RUN_KWARGS)
    # calculate registration quality
    calc_dst_points = (-CMTKtransform(out)).xform(
        landmarks.to_array_and_names(src_landmarks, False)[0]
    )
    dst_points = landmarks.to_array_and_names(dst_landmarks, False)[0]
    return np.sqrt((calc_dst_points - dst_points) ** 2).sum(axis=1).mean()


def do_affine_registration(
    config: RegistrationConfig,
    moving_path: Path,
    fixed_path: Path,
    landmark_affine: Path,
) -> Path:
    """
    Uses CMTK to register affine transform using the landmark affine as the
    initial position
    returns the path of the newly generated xform
    """
    out = config.get_cmtk_transforms_path() / "affine-xform"
    cmtk_command = "registration"
    args = (
        config.cmtk_exe,
        cmtk_command,
        "--threads",
        str(config.ncpu),
        "--initial",
        landmark_affine,
        "--dofs",
        "6,9",
        "--accuracy",
        "0.8",
        "-o",
        out,
        fixed_path,
        moving_path,
    )
    run(args, **RUN_KWARGS)
    return out


# def do_initial_affine(config, moving_path, fixed_path) -> Path:
# out = config.get_cmtk_transforms_path() / "init-affine"
# cmtk_command = "make_initial_affine"
# args = (config.cmtk_exe, cmtk_command, "--threads", str(config.ncpu), "--principal_axes", fixed_path, moving_path, out)
# run(args, **RUN_KWARGS)
# return out


def do_warp_xform(
    config: RegistrationConfig, moving_path: Path, fixed_path: Path, affine: Path
) -> Path:
    """
    Does a warp trasnfrom using CMTK and JRC params
    """
    cmtk_command = "warp"
    out = config.get_cmtk_transforms_path() / "warp-xform"
    args = (
        config.cmtk_exe,
        cmtk_command,
        "--threads",
        str(config.ncpu),
        "-o",
        out,
        "--grid-spacing",
        "80",
        "--fast",
        "--exploration",
        "26",
        "--coarsest",
        "8",
        "--accuracy",
        "0.8",
        "--refine",
        "4",
        "--energy-weight",
        "1e-1",
        "--ic-weight",
        "0",
        "--initial",
        affine,
        fixed_path,
        moving_path,
    )
    run(args, **RUN_KWARGS)
    return out


def apply_registration(
    config: RegistrationConfig,
    moving_path: Path,
    fixed_path: Path,
    xform: Path,
    interpolation: Literal[
        "linear", "nn", "cubic", "pv", "sinc-cosine", "sinc-hamming"
    ] = "linear",
) -> Path:
    """
    Uses CMTK to apply a tranform from image at moving_path in the coordinates
    of image at fixed_path using xform
    returns the path of the newly generated image
    """
    out = config.out_dir / f"{xform.name}-imgs" / moving_path.name
    cmtk_command = "reformatx"
    args = (
        config.cmtk_exe,
        cmtk_command,
        f"--{interpolation}",
        "--outfile",
        out,
        "--floating",
        moving_path,
        fixed_path,
        xform,
    )
    run(args, **RUN_KWARGS)
    return out
