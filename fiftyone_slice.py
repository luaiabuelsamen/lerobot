"""Load the LIBERO depth slice in FiftyOne: the LeRobot dataset natively, plus its point clouds.

    python media_pr/fiftyone_slice.py --serve   # then open http://localhost:5151
"""

import argparse
import json
from pathlib import Path

import fiftyone as fo
import fiftyone.types as fot

HERE = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=HERE / "libero_depth_slice")
    ap.add_argument("--pcd", type=Path, default=HERE / "libero_depth_slice_pcd")
    ap.add_argument("--serve", action="store_true")
    args = ap.parse_args()

    episodes = fo.Dataset.from_dir(dataset_dir=str(args.root), dataset_type=fot.LeRobotDataset,
                                   name="libero-depth-slice", overwrite=True)
    print(episodes)
    print(episodes.first())

    clouds = fo.Dataset("libero-depth-slice-pointclouds", overwrite=True)
    for entry in json.loads((args.pcd / "index.json").read_text()):
        pcd = (args.pcd / entry["pcd"]).resolve()
        scene = fo.Scene(camera=fo.PerspectiveCamera(up="Y"))
        scene.add(fo.PointCloud("cloud", str(pcd), material=fo.PointCloudMaterial(shading_mode="rgb", point_size=2.0)))
        fo3d = pcd.with_suffix(".fo3d")
        scene.write(str(fo3d))
        clouds.add_sample(fo.Sample(filepath=str(fo3d), episode=entry["episode"], frame=entry["frame"]))
    episodes.persistent = True
    clouds.persistent = True
    print(clouds)
    print("cloud sample ids:", clouds.values("id")[:3], "episode ids:", episodes.values("id"))

    if args.serve:
        session = fo.launch_app(clouds, port=5151, remote=True)
        session.wait(-1)


if __name__ == "__main__":
    main()
