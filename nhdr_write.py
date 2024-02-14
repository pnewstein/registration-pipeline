"""
Implements fast simple nhdr writer
"""

from __future__ import annotations
from pathlib import Path
from typing import Any
import sys

import numpy as np
from numpy.typing import NDArray
from skimage import data
import napari
from napari.utils.transforms import Affine


def test_affine():
    blobs = data.binary_blobs(length=64, volume_fraction=0.1, n_dim=3).astype(float)
    viewer = napari.Viewer(ndisplay=3)
    # add the volume
    layer0 = viewer.add_image(
        blobs, scale=[3, 1, 1], name="withp2w", colormap="red", blending="additive"
    )
    layer0.affine = eg_affine_matrix
    layer1 = viewer.add_image(
        blobs, scale=[3, 1, 1], name="two_shot", colormap="green", blending="additive"
    )
    apply_affine_to_data2physical(layer1, eg_affine_matrix)
    comp_affine = layer1._transforms["data2physical"].affine_matrix
    layer2 = viewer.add_image(
        blobs, scale=[1, 1, 1], name="one_shot", colormap="blue", blending="additive"
    )
    apply_affine_to_data2physical(layer2, comp_affine)


def test_order():
    translate = (1, 1, 1)
    scale = (10, 10, 10)
    Affine = napari.utils  # .transforms.Affine
    blobs = data.binary_blobs(length=64, volume_fraction=0.1, n_dim=3).astype(float)
    viewer = napari.Viewer(ndisplay=3)
    layer0 = viewer.add_image(blobs, name="small", blending="additive")
    layer1 = viewer.add_image(blobs, name="ts", colormap="red", blending="additive")
    layer1.translate = (1, 1, 1)
    layer1.scale = (10, 10, 10)
    layer2 = viewer.add_image(blobs, name="tsm", colormap="blue", blending="additive")
    affine = (
        Affine(translate=translate).affine_matrix @ Affine(scale=scale).affine_matrix
    )
    layer2.affine = affine
    layer3 = viewer.add_image(blobs, name="stm", colormap="red", blending="additive")
    affine = (
        Affine(scale=scale).affine_matrix @ Affine(translate=translate).affine_matrix
    )
    layer3.affine = affine
    layer4 = viewer.add_image(blobs, name="sta", colormap="blue", blending="additive")
    layer4.scale = scale
    apply_affine_to_data2physical(layer4, Affine(translate=translate))


DTYPE_MAP: dict[type, str] = {
    np.dtypes.UInt8DType: "uint8",
    np.dtypes.UInt16DType: "uint16",
}  # type ignore

eg_affine_matrix = np.array(
    [
        [-1, 0, 0, 5],
        [0, 2, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ]
)

unit_coords = np.array(
    [
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 1, 1],
    ]
)

axis_inverting_matrix = np.array(
    [[0, 0, 1, 0], [0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1]]
)


def apply_affine_to_data2physical(
    layer: napari.layers.Image, affine_matrix: NDArray[np.float64]
):
    """
    applies an affine matrix to the data2physical transformation layer
    """
    old_affine_matrix = napari.utils.transforms.CompositeAffine(
        scale=layer.scale,
        translate=layer.translate,
        rotate=layer.rotate,
        shear=layer.shear,
        ndim=layer.ndim,
        name=f"affine{layer.name}",
    ).affine_matrix
    assert np.all(old_affine_matrix == layer._transforms["data2physical"])
    affine = napari.utils.transforms.Affine(
        affine_matrix=old_affine_matrix @ affine_matrix
    )
    layer.scale = affine.scale
    layer.translate = affine.translate
    layer.rotate = affine.rotate
    layer.shear = affine.shear
    assert np.all(
        layer._transforms["data2physical"].affine_matrix == affine.affine_matrix
    )


def napari_layer_to_data_affine(
    layer: napari.layers.image.Image,
) -> tuple[np.array, NDArray[np.float64]]:
    """
    returns the data in xyz format and the affine matrix in xyz format
    """
    # reverse data axis order
    assert layer.ndim == 3
    out_data = np.array(layer.data).transpose(2, 1, 0)
    # compose all matrices into affine_matrix
    affine_matrix = napari.utils.transforms.CompositeAffine(
        scale=layer.scale,
        translate=layer.translate,
        rotate=layer.rotate,
        shear=layer.shear,
        ndim=layer.ndim,
        name=f"affine{layer.name}",
    ).affine_matrix
    return out_data, affine_matrix @ axis_inverting_matrix


def format_np_array_triple(triple: np.ndarray | tuple[Any, Any, Any]) -> str:
    """
    formats a numpy array containing 3 elemtens for header
    """
    return f"({triple[0]},{triple[1]},{triple[2]})"


Coords = NDArray[np.float64]  # dim = (3,)


def affine_to_coords(
    affine: NDArray[np.float64],
) -> tuple[Coords, Coords, Coords, Coords]:
    """
    returns origin, x_direction, y_direction, z_direction
    """
    homo_coords = affine @ unit_coords
    coords = homo_coords[:3, :] / homo_coords[3, :]
    origin = coords[:, 0]
    x_direction = coords[:, 1] - origin
    y_direction = coords[:, 2] - origin
    z_direction = coords[:, 3] - origin
    assert len(origin) == len(x_direction) == len(y_direction) == len(z_direction) == 3
    return origin, x_direction, y_direction, z_direction  # type: ignore


def coords_to_affine(
    origin: Coords, x_direction: Coords, y_direction: Coords, z_direction: Coords
) -> NDArray[np.float64]:
    """
    does the inverse of affine_to_coords
    """
    origin_array = np.array(origin)
    homo_coords = np.ones((4, 4))
    homo_coords[:3, 0] = origin_array
    homo_coords[:3, 1] = origin_array + x_direction
    homo_coords[:3, 2] = origin_array + y_direction
    homo_coords[:3, 3] = origin_array + z_direction
    out = homo_coords @ np.linalg.inv(unit_coords)
    # assert that no perspective was invented
    assert np.all(out[-1, :] == [0, 0, 0, 1])
    return out


def fuzz_coors_to_affine():
    random = np.random.uniform(-10000, 10000, size=(10000, 4, 3))
    for rand_2d in random:
        print(coords_to_affine(*rand_2d))
    # also fuzz matrix division on homo cords. I dont quite understand how it works
    homo_coords_s = np.random.uniform(-1000, 1000, size=(1000, 4, 4))
    homo_coords_s[:, -1, :] = 1
    for i, homo_coords in enumerate(homo_coords_s):
        if i == 0:
            continue
        out = homo_coords_s[i] @ np.linalg.inv(homo_coords)
    assert np.all(out[-1, :] == [0, 0, 0, 1])


def write_nhdr(img_data: np.ndarray, affine_matrix: NDArray[np.float64], path: Path):
    """
    writes a 3d image encoded in a numpy array
    """
    if affine_matrix.shape != (4, 4):
        raise TypeError("affine matrix must be a 4 X 4 matrix")
    if type(img_data.dtype) not in DTYPE_MAP:
        raise TypeError("Only np.uint16 and np.uint8 are supported")
    if len(img_data.shape) != 3:
        raise TypeError("Only 3d images are supported")
    # trasnform unit coords
    header_lines: list[str] = []
    header_lines.append("NRRD0004")
    header_lines.append("# made by github.com/pnewstein/nhdr-write")
    data_file = path.with_suffix(".raw")
    header_lines.append("dimension: 3")
    header_lines.append(f"data file: {data_file}")
    header_lines.append("space dimension: 3")
    header_lines.append('space units: "microns" "microns" "microns"')
    origin, x_direction, y_direction, z_direction = affine_to_coords(affine_matrix)
    header_lines.append(f"space origin: {format_np_array_triple(origin)}")
    header_lines.append(
        "space directions: "
        f"{format_np_array_triple(x_direction)} "
        f"{format_np_array_triple(y_direction)} "
        f"{format_np_array_triple(z_direction)}"
    )
    header_lines.append(f"type: {DTYPE_MAP[type(img_data.dtype)]}")
    header_lines.append("encoding: raw")
    order_char = img_data.dtype.byteorder
    if order_char in "=|":
        byte_order = sys.byteorder
    elif order_char == "<":
        byte_order = "little"
    elif order_char == ">":
        byte_order = "big"
    else:
        raise NotImplementedError("unimplemented byte order")
    header_lines.append(f"endian: {byte_order}")
    header_lines.append(f"sizes: {' '.join(str(s) for s in img_data.shape)}")
    header_lines.append("kinds: space space space")
    header_lines.append("\n")
    path.write_text("\n".join(header_lines))
    # write raw numpy array
    if not img_data.flags.c_contiguous:
        img_data = img_data.copy()
        assert img_data.flags.c_contiguous
    with open(data_file, "wb") as f:
        f.write(img_data.data)


def test():
    blobs = data.binary_blobs(length=64, volume_fraction=0.1, n_dim=3).astype(float)
    write_nhdr((blobs * 255).astype(np.uint8), eg_affine_matrix, Path("eg.nhdr"))
