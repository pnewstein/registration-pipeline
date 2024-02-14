from typing import MutableSequence, Sequence

import numpy as np
import itk
from scipy.spatial.transform import Rotation


DIMENSION = 3


def enter_data_into_fixed_point(
    fixed_point: MutableSequence, values_to_add: Sequence[float]
):
    """
    Code that can copy a numpy array into a a Fixed point
    """
    for i, e in enumerate(values_to_add):
        fixed_point[i] = float(e)


def apply_affine(src_points: np.ndarray, affine: np.ndarray) -> np.ndarray:
    """
    with N = number of points to transform
    takes src points shape = (N, 3) and applies affine. returns output points, shape = (N, 3)
    """
    npoints = src_points.shape[0]
    src_homo = np.ones((4, npoints))
    src_homo[:3, :] = src_points.T
    dst_homo = affine @ src_homo
    return dst_homo[:3, :].T


def rigid_landmark_tranform(
    src_points: np.ndarray, dst_points: np.ndarray
) -> np.ndarray:
    """
    returns an affine that can be muliplied by homogenous coordinates to transform
    src vectors into dst vectors
    src and dst points have shape (N, 3) for N = n points
    """
    LandmarkPointType = itk.Point[itk.D, DIMENSION]
    LandmarkContainerType = itk.vector[LandmarkPointType]
    # tell ITK about the fixed and moving landmarks
    moving_landmarks = LandmarkContainerType()
    fixed_landmarks = LandmarkContainerType()
    moving_point = LandmarkPointType()
    fixed_point = LandmarkPointType()
    for src_coord, dst_coord in zip(src_points, dst_points):
        enter_data_into_fixed_point(moving_point, src_coord)
        moving_landmarks.push_back(moving_point)
        enter_data_into_fixed_point(fixed_point, dst_coord)
        fixed_landmarks.push_back(fixed_point)
    # initialize a transform initializer using these data
    TransformInitializerType = itk.LandmarkBasedTransformInitializer[
        itk.Transform[itk.D, DIMENSION, DIMENSION]
    ]
    transform_initializer = TransformInitializerType.New()
    transform_initializer.SetFixedLandmarks(fixed_landmarks)
    transform_initializer.SetMovingLandmarks(moving_landmarks)
    # use the transform initializer to initialize the transform
    transform = itk.VersorRigid3DTransform[itk.D].New()
    transform_initializer.SetTransform(transform)
    transform_initializer.InitializeTransform()
    rotation_matrix = np.array(transform.GetMatrix())
    # compose transformations into first rotate then translate
    inv_rot_mat = np.linalg.inv(rotation_matrix)
    calc_dst = (inv_rot_mat @ src_points.T).T
    subsequent_translate = (dst_points - calc_dst).mean(axis=0)
    # convert to 4X4 affine matrices
    s_translate_affine = np.eye(4)
    s_translate_affine[:3, -1] = subsequent_translate
    inv_rot_mat_affine = np.eye(4)
    inv_rot_mat_affine[:3, :3] = inv_rot_mat
    # compose the two
    return s_translate_affine @ inv_rot_mat_affine


def test_landmark_registration():
    rotation = Rotation.from_euler(seq="xyz", angles=(12, 80, 180), degrees=True)
    src = np.array(
        [
            [0, 0, 30],
            [20, 10, 0],
            [10, 0, 1],
            [50, 2, 10],
            [50, 10, 10],
            [5, -10, 10],
            [6, -10, -5],
            [50, 40, 10],
            [0, 0, 3],
            [20, 1, 0],
            [10, 0, 1],
            [50, 20, 10],
            [50, 15, 10],
            [5, -10, 19],
            [6, -10, -5],
            [50, 46, 10],
        ]
    )
    dst = rotation.apply(src + [100, 500, -5]) + 0 * np.random.normal(size=src.shape)
    affine = rigid_landmark_tranform(src, dst)
    print(score_affine(affine, src, dst))


def score_affine(affine, src, dst) -> float:
    """
    returns the average distance between transformed src and dst
    """
    return np.sqrt((apply_affine(src, affine) - dst) ** 2).sum(axis=1).mean()
