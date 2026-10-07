"""Hero GIF for the depth / point-cloud PRs, rendered locally on CPU.

LIBERO (libero_spatial task 0) with `LiberoEnv(use_depth=True)`: a scripted
pick-and-place drives the arm, and every frame's metric depth and saved
intrinsics go through `DepthToPointCloudStep`. Depth is handed to the step in
millimetres, the unit `LeRobotDataset` returns by default:

* right panel, `depth_scale=1e-3`: the correct cloud;
* far-right panel, the default `depth_scale=1.0` with the step's magnitude check
  disabled, i.e. the behaviour before the guard: every reading is beyond
  `max_depth`, no point is valid, and the step emits a cloud of zeros.

    MUJOCO_GL=egl python media_pr/render_libero_pointcloud.py --out media_pr/libero_pointcloud.gif
"""

import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from libero.libero import benchmark  # noqa: E402
from PIL import Image  # noqa: E402

from lerobot.envs.libero import LiberoEnv  # noqa: E402
from lerobot.processor.depth_processor import OBS_POINTCLOUD, DepthToPointCloudStep  # noqa: E402

CAM = "agentview"
N_POINTS = 8192


class PreGuardStep(DepthToPointCloudStep):
    """The step as it behaved before the magnitude check existed."""

    def _check_scale(self, depth, camera):  # noqa: D102
        return None


def scripted_actions(env):
    """Reach the bowl, grasp, lift, carry over the plate, lower, release."""
    raw = lambda: env._env.env._get_observations()  # noqa: E731
    bowl = raw()["akita_black_bowl_1_pos"].copy()
    plate = raw()["plate_1_pos"].copy()
    waypoints = [
        (bowl + [0, 0, 0.12], -1, 45),
        (bowl + [0, 0, 0.02], -1, 35),
        (bowl + [0, 0, 0.02], 1, 15),
        (bowl + [0, 0, 0.18], 1, 35),
        (plate + [0, 0, 0.18], 1, 50),
        (plate + [0, 0, 0.08], 1, 30),
        (plate + [0, 0, 0.08], -1, 15),
        (plate + [0, 0, 0.20], -1, 25),
    ]
    for target, grip, steps in waypoints:
        for _ in range(steps):
            eef = raw()["robot0_eef_pos"]
            a = np.zeros(7, dtype=np.float32)
            a[:3] = np.clip((target - eef) * 12.0, -1.0, 1.0)
            a[6] = grip
            yield a


def upright_rgb(obs):
    """robosuite RGB is bottom-left origin; flip rows so it lines up with the depth,
    which `LiberoEnv` already puts in top-left origin (`orient_depth`)."""
    return np.ascontiguousarray(obs["pixels"]["image"][::-1])


def to_obs(obs):
    rgb = torch.from_numpy(upright_rgb(obs)).permute(2, 0, 1).float() / 255.0
    depth_mm = torch.from_numpy(np.round(obs["pixels"]["image_depth"] * 1000.0))  # as LeRobotDataset returns it
    return {
        f"observation.images.{CAM}": rgb,
        f"observation.images.{CAM}_depth": depth_mm,
        f"observation.intrinsics.{CAM}": torch.from_numpy(obs["intrinsics"]["image"]),
    }


def plot_cloud(ax, cloud, azim, title, note=None, lim=None):
    ax.cla()
    xyz, rgb = cloud[:, :3], np.clip(cloud[:, 3:6], 0, 1)
    # camera frame (x right, y down, z forward) -> plot (x, z, -y): up is up
    ax.scatter(xyz[:, 0], xyz[:, 2], -xyz[:, 1], c=rgb, s=2.6, depthshade=False, linewidths=0)
    (xl, zl, yl) = lim
    ax.set_xlim(*xl), ax.set_ylim(*zl), ax.set_zlim(*yl)
    ax.set_box_aspect((xl[1] - xl[0], zl[1] - zl[0], yl[1] - yl[0]))
    ax.view_init(elev=24, azim=azim)
    ax.set_axis_off()
    ax.set_title(title, fontsize=11, pad=2)
    if note:
        ax.text2D(0.5, 0.02, note, transform=ax.transAxes, ha="center", fontsize=9.5, color="#b00020")


def box_edges(lim):
    (x0, x1), (z0, z1), (y0, y1) = lim
    c = np.array([[x, z, y] for x in (x0, x1) for z in (z0, z1) for y in (y0, y1)])
    return [(c[i], c[j]) for i in range(8) for j in range(i + 1, 8) if (np.abs(c[i] - c[j]) > 0).sum() == 1]


def plot_collapsed(ax, cloud, azim, lim):
    """Pre-guard output: the scene's box for reference, and where the points actually are."""
    ax.cla()
    for a, b in box_edges(lim):
        ax.plot(*zip(a, b, strict=True), color="#bbbbbb", lw=0.8)
    xyz = cloud[:, :3]
    ax.scatter(xyz[:, 0], xyz[:, 2], -xyz[:, 1], s=60, c="#b00020", depthshade=False)
    ax.text(0, 0, 0.06, " all here", color="#b00020", fontsize=9)
    (xl, zl, yl) = lim
    wl = (min(xl[0], 0) - 0.05, max(xl[1], 0) + 0.05)
    wz = (min(zl[0], 0) - 0.05, max(zl[1], 0) + 0.05)
    wy = (min(yl[0], 0) - 0.05, max(yl[1], 0) + 0.05)
    ax.set_xlim(*wl), ax.set_ylim(*wz), ax.set_zlim(*wy)
    ax.set_box_aspect((wl[1] - wl[0], wz[1] - wz[0], wy[1] - wy[0]))
    ax.view_init(elev=24, azim=azim)
    ax.set_axis_off()
    ax.set_title("before the guard: mm read as m", fontsize=11, pad=2)
    ax.text2D(0.5, 0.02, f"0 valid pixels → all {N_POINTS:,} points at\n"
              "the camera origin. The step now raises.",
              transform=ax.transAxes, ha="center", fontsize=9, color="#b00020")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("media_pr/libero_pointcloud.gif"))
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--fps", type=int, default=12)
    args = ap.parse_args()

    suite = benchmark.get_benchmark_dict()["libero_spatial"]()
    env = LiberoEnv(task_suite=suite, task_id=0, task_suite_name="libero_spatial",
                    camera_name="agentview_image", use_depth=True,
                    observation_height=256, observation_width=256)
    obs, _ = env.reset(seed=0)

    good = DepthToPointCloudStep(num_points=N_POINTS, with_colour=True, depth_scale=1e-3, max_depth=1.6, seed=0)
    bad = PreGuardStep(num_points=N_POINTS, with_colour=True, depth_scale=1.0, max_depth=1.6, seed=0)

    frames_obs = [obs]
    for i, a in enumerate(scripted_actions(env)):
        obs, *_ = env.step(a)
        if i % args.stride == 0:
            frames_obs.append(obs)

    fig = plt.figure(figsize=(10.8, 4.2), dpi=88)
    gs = fig.add_gridspec(2, 3, width_ratios=[0.9, 2.0, 1.25], wspace=0.02, hspace=0.12,
                          left=0.01, right=0.99, top=0.86, bottom=0.02)
    ax_rgb, ax_dep = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[1, 0])
    ax_good = fig.add_subplot(gs[:, 1], projection="3d")
    ax_bad = fig.add_subplot(gs[:, 2], projection="3d")
    fig.suptitle("LIBERO depth + saved intrinsics  →  DepthToPointCloudStep  →  observation.pointcloud",
                 fontsize=12.5, y=0.97)

    stats = {"frames": len(frames_obs), "valid_points_correct": [], "nonzero_points_pre_guard": []}
    images = []
    lim = None
    n = len(frames_obs)
    for k, o in enumerate(frames_obs):
        batch = to_obs(o)
        g = good.observation(dict(batch))[OBS_POINTCLOUD].numpy()
        b = bad.observation(dict(batch))[OBS_POINTCLOUD].numpy()
        depth_m = o["pixels"]["image_depth"]
        n_valid = int(((depth_m > good.min_depth) & (depth_m < good.max_depth)).sum())
        stats["valid_points_correct"].append(n_valid)
        stats["nonzero_points_pre_guard"].append(int((np.abs(b[:, :3]).sum(-1) > 0).sum()))
        if lim is None:
            pts = g[:, :3]
            lo, hi = np.percentile(pts, 1, 0), np.percentile(pts, 99, 0)
            lim = ((lo[0], hi[0]), (lo[2], hi[2]), (-hi[1], -lo[1]))

        ax_rgb.cla(), ax_rgb.imshow(upright_rgb(o)), ax_rgb.set_axis_off()
        ax_rgb.set_title("RGB (agentview)", fontsize=10, pad=2)
        ax_dep.cla(), ax_dep.imshow(depth_m, cmap="turbo", vmin=0.6, vmax=1.6), ax_dep.set_axis_off()
        ax_dep.set_title("depth, 0.6–1.6 m", fontsize=10, pad=2)
        azim = -90 + 38 * np.sin(2 * np.pi * k / n)
        plot_cloud(ax_good, g, azim, f"depth_scale=1e-3 (mm → m): {N_POINTS:,} points", lim=lim)
        plot_collapsed(ax_bad, b, azim, lim)
        fig.canvas.draw()
        images.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
    plt.close(fig)

    # one shared palette keeps the GIF small and flicker-free
    palette = Image.fromarray(np.concatenate(images[:: max(1, len(images) // 8)], 0)).quantize(colors=160)
    pil = [Image.fromarray(im).quantize(palette=palette, dither=Image.Dither.NONE) for im in images]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    pil[0].save(args.out, save_all=True, append_images=pil[1:], duration=int(1000 / args.fps), loop=0, optimize=True)
    stats.update(
        seconds=round(len(pil) / args.fps, 1), bytes=args.out.stat().st_size, task=env.task_description,
        valid_points_correct_min=min(stats["valid_points_correct"]),
        nonzero_points_pre_guard_max=max(stats["nonzero_points_pre_guard"]),
    )
    stats.pop("valid_points_correct"), stats.pop("nonzero_points_pre_guard")
    args.out.with_suffix(".json").write_text(json.dumps(stats, indent=1) + "\n")
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
