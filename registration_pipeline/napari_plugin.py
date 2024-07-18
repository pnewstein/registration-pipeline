"""
This code uses napari to set up registration and view results
"""

from __future__ import annotations
from pathlib import Path
from typing import Callable, Any
from itertools import chain
from copy import deepcopy
import re
import logging

from static_frame import Frame
import napari
import numpy as np
from attrs import define
from qtpy.QtCore import Qt  # type: ignore
from qtpy.QtWidgets import (  # pylint: disable=no-name-in-module
    QComboBox,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)
import nrrd
from nrrd import format_vector
import pandas as pd


from registration_pipeline.landmarks import (
    Landmarks,
    LandmarkInfo,
)
from registration_pipeline.registration_config import (
    RegistrationConfig,
    find_cmtk,
    get_appropriate_image_path,
    TemplateImageInfo,
    OpticLobeCondition,
)

# from registration_pipeline.itk_rigid_landmark import (
# rigid_landmark_tranform,
# score_affine,
# )
from registration_pipeline import wrap_cmtk, landmarks

logger = logging.getLogger("registration pipeline")


def get_dropdowns_callback(
    dropdowns: dict[str, tuple[type, QComboBox]], viewer: napari.viewer.Viewer
) -> Callable:
    """
    dropdowns maps viewer type to a dropdown menu
    """

    def callback(event):
        _ = event
        nonlocal dropdowns
        nonlocal viewer
        for default_layer_name, (layer_type, combo_box) in dropdowns.items():
            text = combo_box.currentText()
            layers: list[napari.layers.Layer] = [
                l for l in viewer.layers if isinstance(l, layer_type)
            ]
            combo_box.clear()
            for layer in layers:
                combo_box.addItem(layer.name)  # type: ignore
            # keep previous name if possible
            layer_names = [l.name for l in layers]
            if text in layer_names:
                combo_box.setCurrentText(text)
            # set default_layer_name if text was not set
            if not text and default_layer_name in layer_names:
                combo_box.setCurrentText(default_layer_name)

    return callback


@define
class Step:  # pylint: disable=too-few-public-methods
    """
    Represents a processing step
    """

    button: QPushButton
    parent: Step | None
    outputs: dict[str, Any]
    state: tuple[napari.layers.Layer, ...]


def get_dropdown_state(
    dropdowns: dict[str, tuple[type, QComboBox]], viewer: napari.viewer.Viewer
) -> tuple[napari.layers.Layer, ...]:
    """
    returns the state of the dropdowns. Same state means that clicking any of
    the Step buttons will have the same output
    """
    return tuple(
        viewer.layers[dropdown[1].currentText()] for dropdown in dropdowns.values()
    )


def gen_colors(n_colors: int):
    """
    generates n_colors number of colors that look
    ok against a gray background

    expected to be ziped with a iterable of length n_colors
    """
    if n_colors == 1:
        yield from ["green"]
    elif n_colors == 2:
        yield from ["green", "magenta"]
    elif n_colors == 3:
        yield from ["green", "red", "blue"]
    else:
        while True:
            yield "gray"


def get_steps(  # pylint: disable=too-many-locals, too-many-statements
    config: RegistrationConfig,
    dropdowns: dict[str, tuple[type, QComboBox]],
    image_scale: np.ndarray,
    viewer: napari.viewer.Viewer,
) -> dict[str, Step]:
    """
    returns a dict that points to steps, which also exist in a tree structure
    """
    # setup landmark affine
    landmark_info_layer_map = {
        li: viewer.layers[dropdowns[li.default_layer_name][1].currentText()]
        for li in landmarks.landmark_infos
    }
    landmark_affine = Step(QPushButton(), None, {}, tuple())

    def landmark_callback():
        """
        callback form of fit_landmark and get_template_image
        """
        nonlocal landmark_affine
        nonlocal landmark_info_layer_map
        nonlocal config
        nonlocal dropdowns
        nonlocal viewer
        logger.info("starting landmark callback")
        landmark_affine_path, optic_lobe_condition = fit_landmark(
            config,
            landmark_info_layer_map,
            dst_landmarks=config.get_landmarks(),
            image_scale=image_scale,
        )
        landmark_affine.outputs["landmark_affine_path"] = landmark_affine_path
        landmark_affine.outputs["optic_lobe_condition"] = optic_lobe_condition
        landmark_affine.outputs["fixed_path"] = get_template_image(
            config, optic_lobe_condition, image_scale
        )
        brp_channel = viewer.layers[dropdowns["brp_chan"][1].currentText()]
        brp_path = save_nhdr(brp_channel, config.out_dir)
        landmark_affine.outputs["moving_path"] = brp_path
        landmark_affine.state = get_dropdown_state(dropdowns, viewer)

    landmark_affine.button.clicked.connect(landmark_callback)

    # setup reformat landmark
    reformat_landmark = Step(
        QPushButton("View landmark transformed"), landmark_affine, {}, tuple()
    )

    def reformat_landmark_callback():
        """
        prepares and calls wrap_cmtk.apply_registration
        """
        nonlocal reformat_landmark
        nonlocal viewer
        nonlocal dropdowns
        logger.info("starting reformat_landmark callback")
        if reformat_landmark.parent is None:
            assert False
        if reformat_landmark.parent.state != get_dropdown_state(dropdowns, viewer):
            reformat_landmark.parent.button.click()
        out = wrap_cmtk.apply_registration(
            config,
            moving_path=reformat_landmark.parent.outputs["moving_path"],
            fixed_path=reformat_landmark.parent.outputs["fixed_path"],
            xform=reformat_landmark.parent.outputs["landmark_affine_path"],
            interpolation="nn",
        )
        load_nhdr(out, viewer, "Landmark xformed")
        reformat_landmark.state = get_dropdown_state(dropdowns, viewer)

    reformat_landmark.button.clicked.connect(reformat_landmark_callback)
    # setup fit affine
    fit_affine = Step(QPushButton(), landmark_affine, {}, tuple())

    def fit_affine_callback():
        """
        prepares and calls wrap_cmtk.do_affine_registration
        """
        nonlocal fit_affine
        nonlocal viewer
        nonlocal dropdowns
        logger.info("starting fit_affine callback")
        if fit_affine.parent is None:
            assert False
        if fit_affine.parent.state != get_dropdown_state(dropdowns, viewer):
            fit_affine.parent.button.click()
        print("done running fit affine parent")
        affine_xform = wrap_cmtk.do_affine_registration(
            config,
            moving_path=fit_affine.parent.outputs["moving_path"],
            fixed_path=fit_affine.parent.outputs["fixed_path"],
            landmark_affine=fit_affine.parent.outputs["landmark_affine_path"],
        )
        fit_affine.state = get_dropdown_state(dropdowns, viewer)
        fit_affine.outputs["affine_xform"] = affine_xform
        fit_affine.outputs["moving_path"] = fit_affine.parent.outputs["moving_path"]
        fit_affine.outputs["fixed_path"] = fit_affine.parent.outputs["fixed_path"]

    fit_affine.button.clicked.connect(fit_affine_callback)
    # setup reformat affine
    reformat_affine = Step(
        QPushButton("View affine transformed"), fit_affine, {}, tuple()
    )

    def reformat_affine_callback():
        """
        prepares and calls wrap_cmtk.apply_registration
        """
        nonlocal reformat_affine
        nonlocal viewer
        nonlocal dropdowns
        logger.info("starting reformat_affine callback")
        if reformat_affine.parent is None:
            assert False
        if reformat_affine.parent.state != get_dropdown_state(dropdowns, viewer):
            reformat_affine.parent.button.click()
        out = wrap_cmtk.apply_registration(
            config,
            moving_path=reformat_affine.parent.outputs["moving_path"],
            fixed_path=reformat_affine.parent.outputs["fixed_path"],
            xform=reformat_affine.parent.outputs["affine_xform"],
            interpolation="nn",
        )
        load_nhdr(out, viewer, "Affine xformed")
        reformat_affine.state = get_dropdown_state(dropdowns, viewer)

    reformat_affine.button.clicked.connect(reformat_affine_callback)
    # setup warp registration
    fit_warp = Step(QPushButton(), fit_affine, {}, tuple())

    def fit_warp_callback():
        """
        prepares and calls wrap_cmtk.do_warp_xform
        """
        nonlocal fit_warp
        nonlocal viewer
        nonlocal dropdowns
        logger.info("starting fit_warp callback")
        if fit_warp.parent is None:
            assert False
        if fit_warp.parent.state != get_dropdown_state(dropdowns, viewer):
            fit_warp.parent.button.click()
        warp_xform = wrap_cmtk.do_warp_xform(
            config,
            moving_path=fit_warp.parent.outputs["moving_path"],
            fixed_path=fit_warp.parent.outputs["fixed_path"],
            affine=fit_warp.parent.outputs["affine_xform"],
        )
        fit_warp.outputs["warp_xform"] = warp_xform
        fit_warp.outputs["moving_path"] = fit_warp.parent.outputs["moving_path"]
        fit_warp.outputs["fixed_path"] = fit_warp.parent.outputs["fixed_path"]
        fit_warp.state = get_dropdown_state(dropdowns, viewer)

    fit_warp.button.clicked.connect(fit_warp_callback)
    # Finaly setup reformat warp
    reformat_warp = Step(QPushButton("View warp transformed"), fit_warp, {}, tuple())

    def reformat_warp_callback():
        """
        prepares and calls wrap_cmtk.apply_registration
        """
        nonlocal reformat_warp
        nonlocal viewer
        nonlocal dropdowns
        logger.info("starting reformat_warp callback")
        if reformat_warp.parent is None:
            assert False
        if reformat_warp.parent.state != get_dropdown_state(dropdowns, viewer):
            reformat_warp.parent.button.click()
        brp_path = reformat_warp.parent.outputs["moving_path"]
        brp_channel = viewer.layers[dropdowns["brp_chan"][1].currentText()]
        warp_img = wrap_cmtk.apply_registration(
            config,
            moving_path=brp_path,
            fixed_path=reformat_warp.parent.outputs["fixed_path"],
            xform=reformat_warp.parent.outputs["warp_xform"],
            interpolation="cubic",
        )
        load_nhdr(warp_img, viewer, "Warped" + brp_channel.name)

        other_channels = [
            l
            for l in viewer.layers
            if isinstance(l, napari.layers.Image)
            and "xformed" not in l.name
            and "Warped" not in l.name
            and l != brp_channel
        ]
        for image_channel, color in zip(
            other_channels, gen_colors(len(other_channels))
        ):
            moving_path = save_nhdr(image_channel, config.out_dir)
            warp_img = wrap_cmtk.apply_registration(
                config,
                moving_path=moving_path,
                fixed_path=reformat_warp.parent.outputs["fixed_path"],
                xform=reformat_warp.parent.outputs["warp_xform"],
                interpolation="cubic",
            )
            load_nhdr(
                warp_img,
                viewer,
                "Warped" + image_channel.name,
                load_kwargs={"blending": "additive", "colormap": color},
            )
        reformat_affine.state = get_dropdown_state(dropdowns, viewer)

    reformat_warp.button.clicked.connect(reformat_warp_callback)
    return {
        "landmark_affine": landmark_affine,
        "reformat_landmark": reformat_landmark,
        "fit_affine": fit_affine,
        "reformat_affine": reformat_affine,
        "fit_warp": fit_warp,
        "reformat_warp": reformat_warp,
    }


class CMTKRegistrar(QWidget):  # pylint: disable=too-few-public-methods
    """
    A wiget that walks you through the registration pipeline
    """

    def __init__(
        self,
        viewer: napari.viewer.Viewer,
        landmark_infos: tuple[LandmarkInfo, ...],
        config: RegistrationConfig,
        image_scale: np.ndarray,
    ):
        super().__init__()
        dropdown_spacing = 2
        self.viewer = viewer
        self.setLayout(QVBoxLayout())
        title_label = QLabel("CMTK Registration")
        title_label.setAlignment(Qt.AlignCenter)
        self.layout().addWidget(title_label)
        # add dropdowns
        self.layout().addSpacing(dropdown_spacing)
        self.dropdowns: dict[str, tuple[type, QComboBox]] = {}
        for landmark_info in landmark_infos:
            self.layout().addSpacing(dropdown_spacing)
            self.layout().addWidget(QLabel(landmark_info.button_label))
            dropdown = QComboBox()
            self.layout().addWidget(dropdown)
            self.layout().addSpacing(dropdown_spacing)
            dropdown.setCurrentText(landmark_info.default_layer_name)
            self.dropdowns[landmark_info.default_layer_name] = (
                napari.layers.Points,
                dropdown,
            )
        self.dropdown_callback = get_dropdowns_callback(self.dropdowns, viewer)
        viewer.layers.events.changed.connect(self.dropdown_callback)
        self.dropdown_callback(None)
        self.layout().addSpacing(dropdown_spacing)
        self.steps = get_steps(config, self.dropdowns, image_scale, viewer)
        self.layout().addWidget(self.steps["reformat_landmark"].button)
        self.layout().addSpacing(dropdown_spacing)
        self.layout().addWidget(QLabel("Brp channel"))
        dropdown = QComboBox()
        self.layout().addWidget(dropdown)
        self.dropdowns["brp_chan"] = (napari.layers.Image, dropdown)
        self.layout().addSpacing(dropdown_spacing)
        self.layout().addWidget(self.steps["reformat_affine"].button)
        self.layout().addSpacing(dropdown_spacing)
        self.layout().addWidget(self.steps["reformat_warp"].button)
        self.dropdown_callback(None)


def get_template_image(
    config: RegistrationConfig,
    optic_lobe_condition: OpticLobeCondition,
    image_scale: np.ndarray,
) -> Path:
    """
    finds the right template image

    image_scale is in zyx
    """
    # must be floats not np.float64
    scale_list = [float(e) for e in image_scale.tolist()]
    out = get_appropriate_image_path(
        config.template_path, 
        TemplateImageInfo(
            z_scale=scale_list[0],
            y_scale=scale_list[1],
            x_scale=scale_list[2],
            optic_lobe_condition=optic_lobe_condition
        )
    )
    return out

class WrongPointCountError(Exception):
    """
    exeption where the wrong number of points are in a layer
    """

    def __init__(self, name: str, expected: int, actual: int):
        super().__init__(f"{name} should have {expected} points, it had {actual}")


def fit_landmark(  # pylint: disable=too-many-locals
    config: RegistrationConfig,
    landmark_info_layer_map: dict[LandmarkInfo, napari.layers.Points],
    dst_landmarks: Landmarks,
    image_scale: np.ndarray,
) -> tuple[Path, OpticLobeCondition]:
    """
    Fits an affine by interpreting the points list
    Raises WrongPointCountError
    """
    # Create options where each bilateral pair is tried each way
    landmark_names = list(
        chain.from_iterable(
            li.landmark_names
            for li in landmarks.landmark_infos
            if "right-optic-lobe" not in li.landmark_names
        )
    )
    src_landmarks_options = [
        pd.DataFrame(np.nan, index=landmark_names, columns=("x", "y", "z"))
    ]
    for landmark_info, layer in landmark_info_layer_map.items():
        scaled_data = layer.data * image_scale
        if landmark_info.landmark_names == ("left-optic-lobe", "right-optic-lobe"):
            continue
        if len(landmark_info.landmark_names) != len(layer.data):
            raise WrongPointCountError(
                layer.name,
                expected=len(landmark_info.landmark_names),
                actual=len(layer.data),
            )
        if len(landmark_info.landmark_names) == 1:
            for landmark_option in src_landmarks_options:
                landmark_option.loc[landmark_info.landmark_names, ["z", "y", "x"]] = (
                    scaled_data
                )
        elif len(landmark_info.landmark_names) == 2:
            # fork src_landmarks_options and revse order for copty
            src_landmarks_options_copy = deepcopy(src_landmarks_options)
            for landmark_option in src_landmarks_options:
                landmark_option.loc[landmark_info.landmark_names, ["z", "y", "x"]] = (
                    scaled_data
                )
            for landmark_option in src_landmarks_options_copy:
                landmark_option.loc[landmark_info.landmark_names, ["z", "y", "x"]] = (
                    scaled_data[::-1, :]
                )
            src_landmarks_options.extend(src_landmarks_options_copy)
        else:
            assert False
    # figure out which landmark is best
    dst_landmarks_to_compare: landmarks.Landmarks = dst_landmarks.loc[
        landmark_names, :
    ]  # type:ignore
    landmark_affine_paths = [
        config.get_cmtk_transforms_path() / f"landmark-{i}/landmark-affine"
        for i, _ in enumerate(src_landmarks_options)
    ]
    score_to_path_landmarks_dict = {
        wrap_cmtk.do_landmark_registration(
            config,
            Frame.from_pandas(src_landmarks),
            dst_landmarks_to_compare,
            path,  # type:ignore
        ): (path, src_landmarks)
        for src_landmarks, path in zip(src_landmarks_options, landmark_affine_paths)
    }
    path, src_landmarks = score_to_path_landmarks_dict[
        min(score_to_path_landmarks_dict.keys())
    ]
    # figure out optic lobe status
    bilateral_landmark_info = next(
        li
        for li in landmarks.landmark_infos
        if len(li.landmark_names) == 2 and "right-optic-lobe" not in li.landmark_names
    )
    optic_lobe_layer = landmark_info_layer_map[
        next(
            li
            for li in landmark_info_layer_map.keys()
            if "right-optic-lobe" in li.landmark_names
        )
    ]
    if len(optic_lobe_layer.data) == 0:
        return path, "none"
    if len(optic_lobe_layer.data) == 2:
        return path, "both"
    if len(optic_lobe_layer.data) == 1:
        # get closest bilateral landmark
        optic_lobe_coords = optic_lobe_layer.data
        bilateral_landmarks = src_landmarks.loc[
            list(bilateral_landmark_info.landmark_names)
        ]
        bilateral_landmarks["distance"] = np.sqrt(
            (bilateral_landmarks.loc[:, ["z", "y", "x"]] - optic_lobe_coords) ** 2
        ).sum(axis=1)
        name = bilateral_landmarks.index[int(bilateral_landmarks["distance"].argmin())]
        if "left" in name:
            return path, "left"
        if "right" in name:
            return path, "right"
        assert False
    raise WrongPointCountError(optic_lobe_layer.name, 2, len(optic_lobe_layer.data))


def test_lanch_pipeline():
    import registration_pipeline

    czi_file = r"/Users/petern/Documents/tmp/20-8-12_39504\ vnd_embryo\ immort_adult_HA,\ V5,\ nc82_slide\ 1.czi".replace(
        "\\", ""
    )
    config = RegistrationConfig(
        template_path=Path().home() / "templates/real_templates/JRC2018_UNISEX",
        cmtk_exe_dir=find_cmtk(),
        out_dir=Path("out0"),
        ncpu=8,
    )
    landmark_infos = registration_pipeline.landmarks.landmark_infos
    image_scale = viewer.layers[2].scale[1:]
    launch_pipeline(viewer, landmark_infos, config)
    from scipy.ndimage import affine_transform

    image = viewer.layers[2].data[0, ...]
    viewer.layers[2].data = image
    scale = np.linalg.inv(np.diag(np.append(viewer.layers[2].scale, 1)))
    combo = np.linalg.inv(affine) @ scale
    out = affine_transform(
        image, combo, order=0, output_shape=(200, 1024, 1024), mode="grid-wrap"
    )
    viewer.add_image(out, colormap="plasma", translate=viewer.layers[0].translate)
    image_layer = viewer.layers[0]
    self = CMTKRegistrar(viewer, landmark_infos, config, image_layer.scale)
    viewer.window.add_dock_widget(self)


def launch_pipeline(
    viewer: napari.viewer.Viewer,
    landmark_infos: tuple[LandmarkInfo, ...],
    config: RegistrationConfig,
):
    """
    initializes napari with the czi file open and
    creates the layers
    """
    image_layers = tuple(l for l in viewer.layers if isinstance(l, napari.layers.Image))
    image_layer = image_layers[0]
    if image_layer.ndim == 4:
        image_layer.data = image_layer.data[0, :]
    add_points_kwargs = {
        "translate": image_layer.translate,
        "scale": image_layer.scale,
        "rotate": image_layer.rotate,
        "affine": image_layer.affine,
        "size": 25,
        "out_of_slice_display": True,
        "ndim": 3,
    }
    for landmark_info in landmark_infos:
        points = viewer.add_points(
            name=landmark_info.default_layer_name, **add_points_kwargs
        )
        points.mode = "add"
    viewer.window.add_dock_widget(
        CMTKRegistrar(viewer, landmark_infos, config, image_layer.scale)
    )


def save_nhdr(
    image_layer: napari.layers.Image, out_dir: Path, file_name: str | Path | None = None
) -> Path:
    """
    saves image layer to path return file saved
    """
    out_dir.mkdir(exist_ok=True)
    if not file_name:
        regex_match = re.match(r"^raw-(.+)-channel$", image_layer.name)
        if regex_match is None:
            fluor: str | None = None
        else:
            fluor = regex_match.group(1)
        scene = image_layer.metadata.get("scene_index")
        if scene is not None and fluor is not None:
            file_name = f"{scene.strip()}-{fluor}.nhdr"
        else:
            file_name = re.sub(r'[<>:"/\\|?*]', "_", image_layer.name) + ".nhdr"
    path = out_dir / file_name
    if image_layer.ndim == 4:
        image_layer.data = image_layer.data[0, :]
    nrrd.write(
        str(path),
        image_layer.data.transpose(2, 1, 0),
        detached_header=True,
        compression_level=0,
    )
    lines = path.read_text("utf-8").split("\n")
    arr1, arr2, arr3 = np.diag(image_layer.scale[::-1])
    lines_to_insert = [
        "space dimension: 3",
        f"space directions: {format_vector(arr1)} {format_vector(arr2)} {format_vector(arr3)}",
        'space units: "microns" "microns" "microns"',
        "kinds: domain domain domain",
        'labels: "x" "y" "z"',
        "",
    ]
    # insert before the first empty line
    for line in lines_to_insert:
        lines.insert(lines.index(""), line)
    path.write_text("\n".join(lines), "utf-8")
    return path


def load_nhdr(
    path: Path,
    viewer: napari.viewer.Viewer,
    layer_name: str | None = None,
    load_kwargs: dict[str, str] | None = None,
) -> napari.layers.Layer:
    """
    loads a nhdr file respecting the scale
    """
    if layer_name is None:
        layer_name = path.with_suffix("").name
    if load_kwargs is None:
        load_kwargs = {}
    data, header = nrrd.read(str(path))

    scale = np.diag(header["space directions"])[::-1]
    logger.info("Adding layer %s", layer_name)
    return viewer.add_image(
        data.transpose(2, 1, 0), scale=scale, rgb=False, name=layer_name, **load_kwargs
    )
