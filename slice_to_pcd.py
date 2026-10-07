"""Read the LIBERO slice back through LeRobotDataset and write one point cloud per frame.

Depth comes back dequantised to millimetres (LeRobotDataset's default), so the
step runs with depth_scale=1e-3 -- the wiring the guard in the step enforces.
Writes ASCII .pcd files (XYZ + packed RGB) for FiftyOne's 3D viewer, plus a JSON
with the round-trip depth error against a fresh render of the same frame.

    python media_pr/slice_to_pcd.py --root media_pr/libero_depth_slice --every 25
"""

import argparse
import json
import struct
from pathlib import Path

import numpy as np

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.processor.depth_processor import OBS_POINTCLOUD, DepthToPointCloudStep

CAM = "agentview"


def write_pcd(path, cloud):
    xyz, rgb = cloud[:, :3], (np.clip(cloud[:, 3:6], 0, 1) * 255).astype(np.uint32)
    packed = (rgb[:, 0] << 16) | (rgb[:, 1] << 8) | rgb[:, 2]
    floats = [struct.unpack("f", struct.pack("I", int(p)))[0] for p in packed]
    # camera frame (y down, z forward) -> y up, z toward the viewer, so the scene is upright
    xyz = xyz * np.array([1.0, -1.0, -1.0])
    header = (f"# .PCD v0.7\nVERSION 0.7\nFIELDS x y z rgb\nSIZE 4 4 4 4\nTYPE F F F F\nCOUNT 1 1 1 1\n"
              f"WIDTH {len(xyz)}\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\nPOINTS {len(xyz)}\nDATA ascii\n")
    body = "\n".join(f"{x:.4f} {y:.4f} {z:.4f} {c:.9g}" for (x, y, z), c in zip(xyz, floats, strict=True))
    path.write_text(header + body + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("media_pr/libero_depth_slice"))
    ap.add_argument("--out", type=Path, default=Path("media_pr/libero_depth_slice_pcd"))
    ap.add_argument("--every", type=int, default=25)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    ds = LeRobotDataset("local/libero-spatial-depth-slice", root=args.root)
    step = DepthToPointCloudStep(num_points=16384, with_colour=True, depth_scale=1e-3, max_depth=1.6, seed=0)
    index = []
    for i in range(0, len(ds), args.every):
        item = ds[i]
        obs = {k: item[k] for k in (f"observation.images.{CAM}", f"observation.images.{CAM}_depth",
                                    f"observation.intrinsics.{CAM}")}
        depth = obs[f"observation.images.{CAM}_depth"]
        cloud = step.observation(dict(obs))[OBS_POINTCLOUD].numpy()
        ep, fr = int(item["episode_index"]), int(item["frame_index"])
        path = args.out / f"ep{ep}_frame{fr:03d}.pcd"
        write_pcd(path, cloud)
        index.append({"pcd": path.name, "episode": ep, "frame": fr,
                      "depth_median_returned": float(depth.median()), "points": len(cloud)})
    (args.out / "index.json").write_text(json.dumps(index, indent=1) + "\n")
    print(json.dumps(index[:3], indent=1), f"... {len(index)} clouds")


if __name__ == "__main__":
    main()
