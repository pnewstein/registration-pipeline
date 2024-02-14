"""
A script that reads all of the landmarks from a points layer on napari
"""

from typing import get_args
from pathlib import Path
import json

import napari
import numpy as np
from registration_pipeline import landmarks

viewer = napari.viewer.Viewer()

names = list(get_args(landmarks.LandmarkName)) + ["nothin"]

text = {
    "string": "label",
    "size": 10,
    "color": "orange",
}

features = {"label": np.empty(0, dtype=int)}

image = viewer.layers[-1]
image.scale = [0.38, 0.38, 0.38]

points = viewer.add_points(
    text=text, features=features, ndim=image.ndim, scale=image.scale
)


@points.events.data.connect
def on_point_add(*args):
    n_points = len(points.data)
    points.feature_defaults["label"] = n_points + 1
    points.properties["label"][0:n_points] = range(1, n_points + 1)
    points.text.values[0:n_points] = names[:n_points]


data = points.data
np.save("points", data)
out_list: list[dict] = []
for i, (zyx, name) in enumerate(zip(data, names)):
    out_list.append(
        landmarks.Landmark(
            xcoord=zyx[2], ycoord=zyx[1], zcoord=zyx[0], name=name
        ).to_json_dict()
    )


Path("JRC2018_UNISEX_38um_iso_16bit-landmarks.json").write_text(
    json.dumps(out_list), "utf-8"
)
