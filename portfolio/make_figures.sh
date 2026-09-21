#!/bin/bash
# Regenerates every generated portfolio figure. Reads the project repos, writes only portfolio/img/.
set -eu
G=/Users/derekwang/Documents/GitHub
OUT="$G/Resume/portfolio/img"
mkdir -p "$OUT"

# Mr. Clean hero: robot docked at the cube table, offscreen, no GUI.
( cd "$G/RL-BOT" && OUT="$OUT" .venv/bin/python - <<'EOF'
import os, mujoco, numpy as np
from PIL import Image
m = mujoco.MjModel.from_xml_path("models/room_scene.xml")
m.vis.global_.offwidth, m.vis.global_.offheight = 2560, 1440
d = mujoco.MjData(m)
mujoco.mj_resetDataKeyframe(m, d, mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "dock_cubes"))
mujoco.mj_forward(m, d)
cam = mujoco.MjvCamera(); mujoco.mjv_defaultCamera(cam)
cam.lookat[:] = d.body("root").xpos + [0, 0, 0.8]
cam.distance, cam.elevation = 2.6, -25
cam.azimuth = float(np.degrees(np.arctan2(cam.lookat[1], cam.lookat[0])))
opt = mujoco.MjvOption(); opt.geomgroup[3] = 0
r = mujoco.Renderer(m, 1440, 2560)
r.update_scene(d, camera=cam, scene_option=opt)
Image.fromarray(r.render()).resize((1600, 900), Image.LANCZOS).save(os.environ["OUT"] + "/mrclean_hero.jpg", quality=88)
EOF
)

# Mr. Clean occupancy grid: obstacles (dark), footprint inflation (grey), planned routes (red).
( cd "$G/RL-BOT" && OUT="$OUT" .venv/bin/python - <<'EOF'
import os, sys
sys.path.insert(0, "src")
from PIL import Image, ImageDraw
from rlbot.navmap import OccupancyGrid
from rlbot.planner import plan, goal_for, Limits
S = 12                                   # pixels per 5 cm cell
raw = OccupancyGrid.from_room(); fat = raw.inflate(0.26)
nx, ny = raw.occupied.shape
img = Image.new("RGB", (nx * S, ny * S), "white"); dr = ImageDraw.Draw(img)
ox, oy = raw.origin; res = raw.resolution
def px(x, y):
    return (((x - ox) / res + 0.5) * S, (ny - ((y - oy) / res + 0.5)) * S)
for i in range(nx):
    for j in range(ny):
        if raw.occupied[i, j]:   c = (40, 40, 40)
        elif fat.occupied[i, j]: c = (205, 205, 205)
        else: continue
        dr.rectangle([i * S, (ny - 1 - j) * S, (i + 1) * S - 1, (ny - j) * S - 1], fill=c)
for name in ("cubes", "ball", "pick"):
    p = plan(raw, (0.0, 0.0, 0.0), goal_for(name), Limits())
    dr.line([px(x, y) for x, y in p.xy], fill=(200, 30, 30), width=4)
img.save(os.environ["OUT"] + "/mrclean_grid.png")
EOF
)

# ACT: frame from the demo video with the MuJoCo GUI panels cropped off.
ffmpeg -v error -y -ss 14 -i "$G/RL-BOT/demo/ACT.mov" -frames:v 1 -vf "crop=1200:1000:535:70" "$OUT/act_frame.jpg"
# Fly-brain point-goal pilot: room view beside the neuron graph.
ffmpeg -v error -y -i "$G/RL-BOT/demo/fly_brain_point_goal_pilot.gif" -vf "select=eq(n\,60)" -frames:v 1 "$OUT/fly_pilot.jpg"

# ACT policy input: the head camera (the two wrist cameras see only the robot's base at the rest pose).
( cd "$G/RL-BOT" && OUT="$OUT" .venv/bin/python - <<'EOF'
import os, mujoco
from PIL import Image
m = mujoco.MjModel.from_xml_path("models/room_scene.xml")
m.vis.global_.offwidth, m.vis.global_.offheight = 1280, 960
d = mujoco.MjData(m)
mujoco.mj_resetDataKeyframe(m, d, mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "dock_ball")); mujoco.mj_forward(m, d)
opt = mujoco.MjvOption(); opt.geomgroup[3] = 0
r = mujoco.Renderer(m, 960, 1280); r.update_scene(d, camera="head_cam", scene_option=opt)
Image.fromarray(r.render()).save(os.environ["OUT"] + "/act_headcam.jpg", quality=88)
EOF
)

# Connectome subgraph at the neurons' real positions (matplotlib lives in the State-Estimation venv).
OUT="$OUT" NPZ="$G/RL-BOT/out/flywire/graph_512.npz" MPLBACKEND=Agg "$G/State-Estimation-And-Tracking/venv/bin/python" - <<'EOF'
import os, numpy as np, matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
z = np.load(os.environ["NPZ"]); P = z["positions"][:, :2]; pre, post = z["pre"], z["post"]
inh = np.isin(z["nt_type"], ["GABA", "GLUT"])
seg = np.stack([P[pre], P[post]], axis=1)
fig, ax = plt.subplots(figsize=(8, 6), dpi=200)
ax.add_collection(LineCollection(seg[~inh[pre]], colors="tab:red",  linewidths=0.2, alpha=0.25))
ax.add_collection(LineCollection(seg[inh[pre]],  colors="tab:blue", linewidths=0.2, alpha=0.25))
ax.scatter(*P.T, s=4, c="0.3", zorder=3)
ax.scatter(*P[z["inputs"]].T,  s=10, c="tab:green",  zorder=4, label="sensory inputs (64)")
ax.scatter(*P[z["outputs"]].T, s=10, c="tab:orange", zorder=4, label="descending outputs (64)")
ax.invert_yaxis(); ax.set_aspect("equal"); ax.axis("off"); ax.legend(loc="lower right", fontsize=7, frameon=False)
fig.savefig(os.environ["OUT"] + "/fly_graph.jpg", bbox_inches="tight", dpi=170, pil_kwargs={"quality": 88})
EOF

# State estimation: one high-resolution frame from the GIF renderer (drive 0001), plus the GPS-dropout figure.
( cd "$G/State-Estimation-And-Tracking" && OUT="$OUT" MPLBACKEND=Agg ./venv/bin/python - <<'EOF'
import os, sys; sys.path.insert(0, "scripts")
from pathlib import Path; import make_gif
FRAME = 100
class Still:
    def __init__(s, fig, update, frames, blit=False): s.fig, s.update = fig, update
    def save(s, out, writer=None, dpi=100): s.update(FRAME); s.fig.savefig(out, dpi=dpi)
make_gif.FuncAnimation = Still
D = Path("data/kitti_raw/2011_09_26/2011_09_26_drive_0001_extract")
make_gif.render(Path("data/cache/pipeline_baseline.npz"), Path(os.environ["OUT"]) / "se_hero_full.png",
                image_dir=D / "image_02", t0_timestamps=D / "oxts" / "timestamps.txt", dpi=200)
EOF
)
sips -Z 1600 "$OUT/se_hero_full.png" --out "$OUT/se_hero.png" >/dev/null && rm "$OUT/se_hero_full.png"
sips -Z 1500 "$G/State-Estimation-And-Tracking/docs/images/failure_gps_dropout.png" --out "$OUT/se_gps_dropout.png" >/dev/null

# WATonomous: the 3D panel of the Foxglove screenshot (costmap, path, rover).
ffmpeg -v error -y -i "$G/wato_rover/assets/demos/demo_nov30.png" -vf "crop=735:700:425:10" "$OUT/wato_foxglove.jpg"

# WATonomous: a detection result saved in the training notebook's outputs (test-set image, conf 0.94).
OUT="$OUT" NB="$G/WATo_rover_yoloinference/wato_rover_inference.ipynb" python3 - <<'EOF'
import os, io, json, base64
from PIL import Image, ImageChops
nb = json.load(open(os.environ["NB"]))
pngs = [o["data"]["image/png"] for o in nb["cells"][7]["outputs"] if "image/png" in o.get("data", {})]
im = Image.open(io.BytesIO(base64.b64decode(pngs[10]))).convert("RGB")
w, h = im.size
im = im.crop((0, int(h * 0.06), w, h))                                   # drop the matplotlib title
box = ImageChops.difference(im, Image.new("RGB", im.size, "white")).getbbox()
im.crop(box).save(os.environ["OUT"] + "/wato_yolo.jpg", quality=88)       # trim white margins
EOF

# arm_dashboard.jpg and arm_dodge.jpg are browser captures of the 7-DOF dashboard (sim_server + npm run dev,
# null-space demo on), taken by hand; they are not regenerated here.

ls -lh "$OUT"
