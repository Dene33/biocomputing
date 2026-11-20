"""Watch the culture drive Asimov 1: live in MuJoCo's viewer, or as a video beside the culture's electrode activity.

    uv run scripts/view.py outputs/reservoir/stand_mea_mcs60_s0                 # live viewer
    uv run scripts/view.py outputs/reservoir/walk_mea_mcs60_s0 --video walk.mp4 # robot + MEA heatmap + raster
    uv run scripts/view.py --teacher --task walk --video teacher.mp4            # the teacher, for comparison
"""

import argparse
import json
import pickle
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from biocompute.loop import run_episode
from biocompute.robot.asimov import AsimovEnv
from biocompute.robot.teacher import Teacher
from biocompute.train import dagger
from biocompute.train.setup import Setup, build
from biocompute.train.tasks import make_task

parser = argparse.ArgumentParser()
parser.add_argument("run", nargs="?", help="A train_reservoir.py output folder")
parser.add_argument("--teacher", action="store_true", help="Drive with the teacher policy instead")
parser.add_argument("--task", default=None)
parser.add_argument("--seconds", type=float, default=20.0)
parser.add_argument("--video", default=None, help="Write an MP4 instead of opening the viewer")
parser.add_argument("--fps", type=int, default=25)
parser.add_argument("--seed", type=int, default=100)
args = parser.parse_args()
torch.set_num_threads(4)

env, teacher = AsimovEnv(), Teacher()
if args.teacher:
    task = make_task(args.task or "walk", args.seconds, seed=args.seed)

    class TeacherCtrl:
        device = None
        last_block = None

        def reset_readout_state(self):
            pass

        def act(self, obs):
            return teacher(obs), np.zeros(1)

    ctrl, title = TeacherCtrl(), "teacher policy"
else:
    summary = json.loads((Path(args.run) / "summary.json").read_text())
    task = make_task(args.task or summary["args"]["task"], args.seconds, seed=args.seed)
    setup = Setup(**{k: tuple(v) if isinstance(v, list) else v for k, v in summary["setup"].items()})
    sa = summary["args"]
    if "workers" in sa:  # train_dagger.py: the encoder was fitted on teacher + DART episodes
        obs = dagger.teacher_observations(env, teacher, [make_task(sa["task"], sa["seconds"], seed=sa["seed"])],
                                          seconds=sa["seconds"], episodes=6, noise=0.3, seed=sa["seed"])
    else:
        obs = dagger.teacher_observations(env, teacher, [make_task(sa["task"], args.seconds)], args.seconds)
    lo = (env.run.action_clip[:, 0] - env.run.default_joint_pos) / env.run.action_scale
    hi = (env.run.action_clip[:, 1] - env.run.default_joint_pos) / env.run.action_scale
    ctrl = build(setup, obs, np.clip(np.array([teacher(o) for o in obs]), lo, hi))
    with open(Path(args.run) / "readout.pkl", "rb") as f:
        ctrl.readout = pickle.load(f)
    dagger.warm_up(ctrl)
    title = f"culture ({setup.wells} wells, {setup.mea})" if setup.wells > 1 else f"culture ({setup.source}, {setup.mea})"

frames_every = max(1, int(round(1.0 / (args.fps * env.dt))))

if args.video is None:
    import mujoco.viewer

    env.reset()
    viewer = mujoco.viewer.launch_passive(env.robot.model, env.robot.data)

    def on_step(step, env_, ctrl_):
        if step % frames_every == 0:
            viewer.cam.lookat[:] = env_.root_pos
            viewer.sync()
            time.sleep(max(0.0, frames_every * env_.dt - 0.005))

    # run_episode resets the env, which builds a new MuJoCo model; re-launch the viewer on it.
    _reset = env.reset

    def reset(*a, **k):
        global viewer
        _reset(*a, **k)
        viewer.close()
        viewer = mujoco.viewer.launch_passive(env.robot.model, env.robot.data)

    env.reset = reset
    ep = run_episode(env, ctrl, task.seconds, task.command_fn, push_fn=task.push_fn, on_step=on_step,
                     stop_on_fall=False)
    print(f"fell_at={ep.fell_at:.2f}, x={ep.root_pos[-1][0]:.2f} m")
    viewer.close()
else:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import mujoco

    W, H = 640, 480
    frames, mea_counts, rasters = [], [], []

    def on_step(step, env_, ctrl_):
        block = getattr(ctrl_, "last_block", None)
        if block is not None:
            mea_counts.append(block.counts(1)[0])
            rasters.append((step, block.channels.copy(), block.times_ms.copy()))
        if step % frames_every == 0:
            renderer.update_scene(env_.robot.data, camera=cam)
            cam.lookat[:] = env_.root_pos
            pushing = bool(env_.push_steps_left) or bool(np.any(env_.data.xfrc_applied[env_.pelvis, :2]))
            frames.append((step, renderer.render().copy(), len(mea_counts), pushing))

    env.reset()
    renderer = mujoco.Renderer(env.robot.model, H, W)
    cam = mujoco.MjvCamera()
    cam.distance, cam.azimuth, cam.elevation = 2.5, 135.0, -15.0
    _reset = env.reset

    def reset(*a, **k):
        global renderer
        _reset(*a, **k)
        renderer = mujoco.Renderer(env.robot.model, H, W)

    env.reset = reset
    ep = run_episode(env, ctrl, task.seconds, task.command_fn, push_fn=task.push_fn, on_step=on_step,
                     stop_on_fall=False)
    print(f"fell_at={ep.fell_at:.2f}, x={ep.root_pos[-1][0]:.2f} m; rendering {len(frames)} frames")

    device = getattr(ctrl, "device", None)
    xy = device.mea.xy.copy() if device is not None else None
    raster_channels = device.n_channels if device is not None else 0
    if device is not None and device.mea.wells > 1:
        # Wells sit 10 mm apart in the model; tile them 3 per row for the picture.
        per = device.mea.channels_per_well
        span = np.ptp(xy[:per, 0]) * 1.25
        for k in range(device.mea.wells):
            sl = slice(k * per, (k + 1) * per)
            xy[sl, 0] = xy[sl, 0] - k * device.mea.cfg.well_spacing_um + (k % 3) * span
            xy[sl, 1] = xy[sl, 1] - (k // 3) * span
        raster_channels = per  # raster: the first well only
    fig = plt.figure(figsize=(12.8, 4.8), dpi=100)
    proc = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s",
                             "1280x480", "-r", str(args.fps), "-i", "-", "-pix_fmt", "yuv420p", args.video],
                            stdin=subprocess.PIPE)
    counts = np.array(mea_counts) if mea_counts else None
    for step, img, n_blocks, pushing in frames:
        fig.clf()
        ax0 = fig.add_axes([0, 0, 0.5, 1])
        ax0.imshow(img)
        ax0.axis("off")
        ax0.set_title(f"{title}  t={step * env.dt:4.1f}s" + ("   PUSH" if pushing else ""), y=0.93,
                      color="r" if pushing else "w", fontsize=11)
        if counts is not None and n_blocks:
            recent = counts[max(0, n_blocks - 5):n_blocks].sum(0)
            ax1 = fig.add_axes([0.53, 0.55, 0.2, 0.4])
            dot = 120 if len(xy) <= 64 else max(2, 6000 // len(xy))
            ax1.scatter(xy[:, 0], xy[:, 1], c=recent, s=dot, cmap="magma", vmin=0,
                        vmax=max(4, np.percentile(counts, 99)))
            stim = ctrl.encoder.channels
            ax1.scatter(xy[stim, 0], xy[stim, 1], s=4 if len(xy) > 64 else 12, c="c", marker="x", linewidths=0.5)
            ax1.set_title("MEA: spikes in last 100 ms (x = sensory stim)", fontsize=9)
            ax1.set_aspect("equal")
            ax1.axis("off")
            ax2 = fig.add_axes([0.53, 0.08, 0.45, 0.4])
            for s, ch, tt in rasters[max(0, n_blocks - 100):n_blocks]:
                keep = ch < raster_channels
                ax2.scatter(s * env.dt + tt[keep] / 1000.0, ch[keep], s=0.3, c="k")
            ax2.set_xlim(max(0, step * env.dt - 2.0), max(2.0, step * env.dt))
            ax2.set_ylim(-1, raster_channels)
            ax2.set_ylabel("channel" if raster_channels == device.n_channels else "channel (well 1)")
            ax2.set_xlabel("s")
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())[..., :3]
        proc.stdin.write(np.ascontiguousarray(buf).tobytes())
    proc.stdin.close()
    proc.wait()
    print(f"video: {args.video}")
