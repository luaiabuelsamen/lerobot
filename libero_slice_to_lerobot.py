"""Record a small LIBERO RGB + depth slice into a real LeRobotDataset, locally.

Two scripted pick-and-place episodes of libero_spatial task 0 through
`LiberoEnv(use_depth=True)`, written with the ordinary `LeRobotDataset.create` /
`add_frame` / `save_episode` path: RGB video, depth as a 12-bit depth video
(`is_depth_map`), and the per-frame intrinsics LeRobot used to discard.

    MUJOCO_GL=egl python media_pr/libero_slice_to_lerobot.py --root media_pr/libero_depth_slice
"""

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
from libero.libero import benchmark

from lerobot.configs.video import DepthEncoderConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.envs.libero import LiberoEnv

sys.path.insert(0, str(Path(__file__).parent))
from render_libero_pointcloud import scripted_actions, upright_rgb  # noqa: E402

CAM = "agentview"
SIZE = 256


def features():
    return {
        "observation.state": {"dtype": "float32", "shape": (8,), "names": None},
        "action": {"dtype": "float32", "shape": (7,), "names": None},
        f"observation.images.{CAM}": {"dtype": "video", "shape": (SIZE, SIZE, 3),
                                      "names": ["height", "width", "channels"]},
        f"observation.images.{CAM}_depth": {"dtype": "video", "shape": (SIZE, SIZE, 1),
                                            "names": ["height", "width", "channels"],
                                            "info": {"is_depth_map": True}},
        f"observation.intrinsics.{CAM}": {"dtype": "float32", "shape": (3, 3), "names": ["row", "col"]},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("media_pr/libero_depth_slice"))
    ap.add_argument("--episodes", type=int, default=2)
    ap.add_argument("--stride", type=int, default=2)
    args = ap.parse_args()
    if args.root.exists():
        shutil.rmtree(args.root)

    suite = benchmark.get_benchmark_dict()["libero_spatial"]()
    ds = LeRobotDataset.create(repo_id="local/libero-spatial-depth-slice", fps=10, features=features(),
                               root=args.root, robot_type="panda", use_videos=True,
                               depth_encoder=DepthEncoderConfig(depth_min=0.3, depth_max=4.0))
    for ep in range(args.episodes):
        env = LiberoEnv(task_suite=suite, task_id=0, task_suite_name="libero_spatial",
                        camera_name="agentview_image", use_depth=True, init_states=True,
                        observation_height=SIZE, observation_width=SIZE)
        env.init_state_id = ep
        obs, _ = env.reset(seed=ep)
        for i, a in enumerate(scripted_actions(env)):
            if i % args.stride == 0:
                raw = env._env.env._get_observations()
                state = np.concatenate([raw["robot0_eef_pos"], raw["robot0_eef_quat"],
                                        raw["robot0_gripper_qpos"][:1]]).astype(np.float32)
                ds.add_frame({
                    "observation.state": state,
                    "action": a.astype(np.float32),
                    f"observation.images.{CAM}": upright_rgb(obs),
                    f"observation.images.{CAM}_depth": np.ascontiguousarray(obs["pixels"]["image_depth"])[..., None],
                    f"observation.intrinsics.{CAM}": obs["intrinsics"]["image"].astype(np.float32),
                    "task": env.task_description,
                })
            obs, *_ = env.step(a)
        ds.save_episode()
        env.close()
    ds.finalize()
    print(f"wrote {ds.num_episodes} episodes, {ds.num_frames} frames to {args.root}")


if __name__ == "__main__":
    main()
