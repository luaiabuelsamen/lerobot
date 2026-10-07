# Media for the DP3 / depth PRs

Not code for upstream: images referenced from the PR descriptions of the
`split/4696-*` branches, and the scripts that made them. Everything was rendered
locally on CPU (MuJoCo offscreen + matplotlib) on a Jetson Orin.

| file | made by |
|---|---|
| `libero_pointcloud.gif` (+ `.json` stats) | `render_libero_pointcloud.py` |
| `fiftyone_lerobot_episode.png`, `fiftyone_pointcloud.png` | `libero_slice_to_lerobot.py` -> `slice_to_pcd.py` -> `fiftyone_slice.py` (FiftyOne 1.22.1) |
