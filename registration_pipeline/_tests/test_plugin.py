"""
Some tests for the plugin portion of the pipeline
"""

from typing import get_args
from pathlib import Path
from typing import get_args
import shutil
from tempfile import TemporaryDirectory
import logging

import numpy as np
import napari
from xform import CMTKtransform
from qtpy.QtWidgets import QComboBox, QWidget
from attrs import evolve

from registration_pipeline import landmarks, napari_plugin, registration_config

logger = logging.getLogger("registration pipeline")
logging.basicConfig(level="INFO")

DATA_DIR = Path(__file__).parent / "data"

config = registration_config.RegistrationConfig(
    template_path=DATA_DIR / "template.nhdr",
    cmtk_exe_dir=registration_config.find_cmtk(),
    out_dir=DATA_DIR,
    ncpu=8,
)

src_landmarks = landmarks.from_array_and_names(
    np.array(
        [
            [2, 2, 2],
            [4, 5, 2],
            [2, 2, 8],
            [5, 8, 5],
            [2, 3, 2],
        ]
    ),
    [
        "left-protocerebral-bridge",
        "right-protocerebral-bridge",
        "elipsoid-body",
        "between-antenal-lobes",
        "left-optic-lobe",
    ],
    False,
)


def get_image_from_coords(lmarks: landmarks.Landmarks) -> np.ndarray:
    src_coords, _ = landmarks.to_array_and_names(lmarks, False)
    src_image = np.zeros((10, 11, 12), np.ubyte)
    src_image[src_coords[:, 0], src_coords[:, 1], src_coords[:, 2]] = 255
    return src_image


def get_rand_landmarks() -> landmarks.Landmarks:
    names = get_args(landmarks.LandmarkName)
    np.random.randint(0, 9, (len(names), 3))
    return landmarks.from_array_and_names(
        np.random.randint(0, 9, size=(len(names), 3)), names, False
    )


def make_test_template(template_path: Path):
    template_path.mkdir(exist_ok=True)
    eg_landmarks = get_rand_landmarks()
    eg_landmarks.to_csv(template_path / "landmarks.csv")
    image = get_image_from_coords(eg_landmarks)
    layer = napari.layers.Image(data=image)
    napari_plugin.save_nhdr(
        layer, template_path, file_name="JRC2018_UNISEX_20x_gen1.nhdr"
    )
    napari_plugin.save_nhdr(layer, template_path, file_name="JRC2018_UNISEX.nhdr")


def get_dropdowns(viewer: napari.viewer.Viewer) -> dict[str, tuple[type, QComboBox]]:
    src_landmarks = get_rand_landmarks()
    dropdowns: dict[str, tuple[type, QComboBox]] = {}
    for landmark_info in landmarks.landmark_infos:
        dropdown = QComboBox()
        dropdown.addItem(landmark_info.default_layer_name)
        dropdowns[landmark_info.default_layer_name] = (
            napari.layers.Points,
            dropdown,
        )
        rand_points_data = src_landmarks.loc[
            list(landmark_info.landmark_names), ["x", "y", "z"]
        ]
        viewer.add_points(name=landmark_info.default_layer_name, data=rand_points_data)
    dropdown = QComboBox()
    dropdowns["brp_chan"] = (napari.layers.Image, dropdown)
    dropdown.addItem("S00 AF647-T2")
    viewer.add_image(name="S00 AF647-T2", data=get_image_from_coords(src_landmarks))
    return dropdowns


def test_fit_affine():
    src_coords, names = landmarks.to_array_and_names(src_landmarks, False)
    dst_coords = src_coords + [1, -2, 1]
    dst_landmarks = landmarks.from_array_and_names(
        dst_coords, src_landmarks.index, False
    )
    landmark_info_dict: dict[landmarks.LandmarkInfo, napari.layers.Points] = {}
    for landmark_info in landmarks.landmark_infos:
        names_mask = np.isin(src_landmarks.index, landmark_info.landmark_names)
        data_points = src_landmarks.loc[names_mask, ["z", "y", "x"]]
        landmark_info_dict[landmark_info] = napari.layers.Points(data=data_points)
    path, optic_lobe_condition = napari_plugin.fit_landmark(
        config, landmark_info_dict, dst_landmarks, np.array([1, 1, 1])
    )
    assert optic_lobe_condition == "left"
    transform = -CMTKtransform(path)
    assert np.allclose(transform.xform(src_coords), dst_coords)


def test_get_dropdown_state(make_napari_viewer):
    viewer = make_napari_viewer()
    dropdowns = get_dropdowns(viewer)
    state = napari_plugin.get_dropdown_state(dropdowns, viewer)
    assert state == napari_plugin.get_dropdown_state(dropdowns, viewer)
    viewer.add_points(name="test_points")
    _, button = dropdowns[landmarks.landmark_infos[0].default_layer_name]
    button.addItem("test_points")
    button.setCurrentText("test_points")
    print(state)
    diff_state = napari_plugin.get_dropdown_state(dropdowns, viewer)
    print(diff_state)
    assert diff_state != state


def test_steps(make_napari_viewer):
    viewer = make_napari_viewer()
    dropdowns = get_dropdowns(viewer)
    with TemporaryDirectory() as temp_dir_str:
        temp_dir = Path(temp_dir_str)
        print(temp_dir)
        make_test_template(temp_dir)
        test_config = evolve(config, template_path=temp_dir, out_dir=temp_dir)
        steps = napari_plugin.get_steps(
            test_config, dropdowns, np.array([1, 1, 1]), viewer
        )
        steps["reformat_warp"].button.click()
        assert sum(1 for _ in temp_dir.glob("**/*")) == 29
        image_channels = [
            l for l in viewer.layers if isinstance(l, napari.layers.Image)
        ]
        assert len(image_channels) == 2

    with TemporaryDirectory() as temp_dir_str:
        temp_dir = Path(temp_dir_str)
        make_test_template(temp_dir)
        test_config = evolve(config, template_path=temp_dir, out_dir=temp_dir)
        steps = napari_plugin.get_steps(
            test_config, dropdowns, np.array([1, 1, 1]), viewer
        )
        steps["reformat_affine"].button.click()
        assert sum(1 for _ in temp_dir.glob("**/*")) == 24

    with TemporaryDirectory() as temp_dir_str:
        temp_dir = Path(temp_dir_str)
        make_test_template(temp_dir)
        test_config = evolve(config, template_path=temp_dir, out_dir=temp_dir)
        steps = napari_plugin.get_steps(
            test_config, dropdowns, np.array([1, 1, 1]), viewer
        )
        steps["reformat_landmark"].button.click()
        assert sum(1 for _ in temp_dir.glob("**/*")) == 19


def test_load_nhdr(make_napari_viewer):
    # first write sample data
    viewer = make_napari_viewer()
    test_layer = napari.layers.Image(
        data=np.arange(3**3).reshape((3, 3, 3)), scale=[1, 2, 3], rgb=False
    )
    napari_plugin.save_nhdr(test_layer, DATA_DIR, "small.nhdr")
    napari_plugin.load_nhdr(DATA_DIR / "small.nhdr", viewer)
    out_layer = viewer.layers[0]
    print(out_layer)
    print(out_layer.data)
    print(out_layer.scale)


if __name__ == "__main__":
    # test_fit_affine()
    test_steps(napari.viewer.Viewer)
