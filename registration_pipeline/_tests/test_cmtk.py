"""
tests cmtk interface
"""

from typing import get_args
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from scipy.spatial.transform import Rotation
from xform import CMTKtransform
import napari
import shutil
import nrrd
from attrs import evolve

from registration_pipeline import wrap_cmtk, landmarks, registration_config
from registration_pipeline.napari_plugin import save_nhdr


DATA_DIR = Path(__file__).parent / "data"

SRC_LANDMARK_PATH = DATA_DIR / "src_landmarks"
DST_LANDMARK_PATH = DATA_DIR / "dst_landmarks"

src_landmarks = landmarks.from_array_and_names(
    np.array(
        [
            [2, 2, 2],
            [4, 5, 2],
            [2, 2, 8],
            [5, 8, 5],
        ]
    ),
    get_args(landmarks.LandmarkName)[:4],
    False,
)

config = registration_config.RegistrationConfig(
    template_path=DATA_DIR / "template.nhdr",
    cmtk_exe_dir=registration_config.find_cmtk(),
    out_dir=DATA_DIR,
    ncpu=8,
)

zyx = False


def test_landmark_translate():
    src_coords, names = landmarks.to_array_and_names(src_landmarks, zyx)
    dst_coords = src_coords + [1, -2, 1]
    dst_landmarks = landmarks.from_array_and_names(dst_coords, src_landmarks.index, zyx)
    landmark_affine = Path(DATA_DIR / "landmark-xform")
    wrap_cmtk.do_landmark_registration(
        config, src_landmarks, dst_landmarks, landmark_affine
    )
    transform = -CMTKtransform(landmark_affine)
    assert np.allclose(transform.xform(src_coords), dst_coords)


def test_landmark_rotate():
    src_coords, names = landmarks.to_array_and_names(src_landmarks, zyx)
    dst_coords = Rotation.from_euler("XYZ", (10, 80, -100), degrees=True).apply(
        src_coords
    ) + [-2, 7, 9]
    dst_landmarks = landmarks.from_array_and_names(dst_coords, src_landmarks.index, zyx)
    landmark_affine = Path(DATA_DIR / "landmark-xform")
    wrap_cmtk.do_landmark_registration(
        config, src_landmarks, dst_landmarks, landmark_affine
    )
    transform = -CMTKtransform(landmark_affine)
    assert np.allclose(transform.xform(src_coords), dst_coords)


# viewer = napari.viewer.Viewer()
def test_image_translate():
    src_coords, _ = landmarks.to_array_and_names(src_landmarks, zyx)
    dst_coords = src_coords + [1, 1, -1]
    dst_landmarks = landmarks.from_array_and_names(dst_coords, src_landmarks.index, zyx)
    landmark_affine = Path(DATA_DIR / "landmark-xform")
    wrap_cmtk.do_landmark_registration(
        config, src_landmarks, dst_landmarks, landmark_affine
    )
    src_coords_xyz = src_coords[:, ::-1] if zyx else src_coords
    src_image = np.zeros((10, 11, 12), np.ubyte)
    src_image[src_coords_xyz[:, 0], src_coords_xyz[:, 1], src_coords_xyz[:, 2]] = 255
    for src_coord in src_coords:
        assert src_image[src_coord[0], src_coord[1], src_coord[2]] == 255

    src_layer = napari.layers.Image(
        data=src_image.transpose(2, 1, 0), blending="additive", colormap="red"
    )
    dst_image = np.zeros((10, 11, 12), np.ubyte)
    dst_coords_int = np.round(dst_coords).astype(int)
    dst_coords_xyz = dst_coords_int[:, ::-1] if zyx else dst_coords_int
    dst_image[dst_coords_xyz[:, 0], dst_coords_xyz[:, 1], dst_coords_xyz[:, 2]] = 255
    dst_layer = napari.layers.Image(
        data=dst_image.transpose(2, 1, 0), blending="additive", colormap="green"
    )
    save_nhdr(src_layer, DATA_DIR, file_name="test_src.nhdr")
    save_nhdr(dst_layer, DATA_DIR, file_name="test_dst.nhdr")
    out = wrap_cmtk.apply_registration(
        config,
        DATA_DIR / "test_src.nhdr",
        DATA_DIR / "test_src.nhdr",
        landmark_affine,
        "nn",
    )
    img, _ = nrrd.read(str(out))
    assert (
        np.min(img[dst_coords_xyz[:, 0], dst_coords_xyz[:, 1], dst_coords_xyz[:, 2]])
        == 255
    )


def test_image_rotate():
    src_coords, names = landmarks.to_array_and_names(src_landmarks, zyx)
    dst_coords = Rotation.from_euler("XYZ", (10, 80, -100), degrees=True).apply(
        src_coords
    ) + [-2, 7, 9]
    dst_landmarks = landmarks.from_array_and_names(dst_coords, src_landmarks.index, zyx)
    landmark_affine = Path(DATA_DIR / "landmark-xform")
    score = wrap_cmtk.do_landmark_registration(
        config, src_landmarks, dst_landmarks, landmark_affine
    )
    assert score < 0.01
    src_coords_xyz = src_coords[:, ::-1] if zyx else src_coords
    src_image = np.zeros((10, 11, 12), np.ubyte)
    src_image[src_coords_xyz[:, 0], src_coords_xyz[:, 1], src_coords_xyz[:, 2]] = 255
    src_layer = napari.layers.Image(
        data=src_image.transpose(2, 1, 0), blending="additive", colormap="red"
    )
    dst_image = np.zeros((10, 11, 12), np.ubyte)
    dst_coords_int = np.round(dst_coords).astype(int)
    dst_coords_xyz = dst_coords_int[:, ::-1] if zyx else dst_coords_int
    dst_image[dst_coords_xyz[:, 0], dst_coords_xyz[:, 1], dst_coords_xyz[:, 2]] = 255
    dst_layer = napari.layers.Image(
        data=dst_image.transpose(2, 1, 0), blending="additive", colormap="green"
    )
    save_nhdr(src_layer, DATA_DIR, file_name="test_src.nhdr")
    save_nhdr(dst_layer, DATA_DIR, file_name="test_dst.nhdr")
    out = wrap_cmtk.apply_registration(
        config,
        DATA_DIR / "test_src.nhdr",
        DATA_DIR / "test_src.nhdr",
        landmark_affine,
        "nn",
    )
    img, _ = nrrd.read(str(out))
    assert (
        np.min(img[dst_coords_xyz[:, 0], dst_coords_xyz[:, 1], dst_coords_xyz[:, 2]])
        == 255
    )
    # wrap_cmtk.apply_registration(config, Path('tests/src_image.nhdr'), DATA_DIR / "test_src.nhdr", landmark_affine)


def interpolate(image: napari.layers.Image, new_scale: np.ndarray, out_dir: Path):
    """
    interpolates image to new scale
    """
    new_n_pix = np.round(image.data.shape * image.scale / new_scale).astype(int)
    dummy_dst = np.zeros(new_n_pix, np.uint8)
    dummy_img = napari.layers.Image(dummy_dst, scale=new_scale)
    with TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)
        tmp_config = evolve(config, out_dir=temp_dir)
        save_nhdr(image, temp_dir, file_name="src_image.nhdr")
        save_nhdr(dummy_img, temp_dir, file_name="dummy_src.nhdr")
        empty_affine = Path(temp_dir / "empty_affine")
        wrap_cmtk.do_landmark_registration(
            tmp_config, src_landmarks, src_landmarks, empty_affine
        )
        wrap_cmtk.apply_registration(
            tmp_config,
            temp_dir / "src_image.nhdr",
            temp_dir / "dummy_src.nhdr",
            empty_affine,
        )
        shutil.move(
            temp_dir / "landmark-xform-imgs/src_image.nhdr", out_dir / "src_image.nhdr"
        )
        shutil.move(
            temp_dir / "landmark-xform-imgs/src_image.raw.gz",
            out_dir / "src_image.raw.gz",
        )


if __name__ == "__main__":
    test_image_rotate()
