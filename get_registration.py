from typing import MutableSequence, Sequence

import numpy as np
import itk
from scipy.spatial.transform import Rotation

rotation = Rotation.from_euler(seq="xyz", angles=(10, 0, 180), degrees=True)

DIMENSION = 3

src = np.array([
    [0, 0, 30],
    [20, 10, 0],
    [10, 0, 1],
    [10, 0, 10],
    [10, 20, 10],
    [10, 0, 30],
    [10, -10, 10],
])

dst = rotation.apply(src + [10, 5, -5])


def enter_data_into_fixed_point(fixed_point: MutableSequence, values_to_add: Sequence[float]):
    for i, e in enumerate(values_to_add):
        fixed_point[i] = float(e)

def rigid_landmark_tranform(src_points: np.ndarray, dst_points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    returns a matrix and a translation array
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
    return np.array(transform.GetMatrix()), np.array(transform.GetTranslation())

mat, trans = rigid_landmark_tranform(src, dst)
print(Rotation.from_matrix(mat).inv().apply(src)-trans)
print(dst)

# out, _ = cv2.estimateAffine3D(src, dst, force_rotation=True)
# print(out)
