import os
import glob
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import requests
from PIL import Image
from io import BytesIO
import cv2

# ----------------------------------------
# Parse command-line arguments
# ----------------------------------------
parser = argparse.ArgumentParser(
    description="Visualize drone flight logs and synchronized RGB/IR imagery"
)
parser.add_argument(
    "data_dir", help="Directory containing flight data (CSV, TXT, RGB/IR folders)"
)
parser.add_argument(
    "--rgb-offset", type=int, default=0, help="Offset to apply to RGB trigger indices"
)
parser.add_argument(
    "--ir-offset", type=int, default=0, help="Offset to apply to IR trigger indices"
)
parser.add_argument(
    "--rgb-rotate", type=float, default=0.0, help="Degrees to rotate each RGB frame"
)
parser.add_argument(
    "--ir-rotate", type=float, default=0.0, help="Degrees to rotate each IR frame"
)
args = parser.parse_args()

data_dir = args.data_dir
rgb_offset = args.rgb_offset
ir_offset = args.ir_offset
rgb_angle = args.rgb_rotate
ir_angle = args.ir_rotate

# ----------------------------------------
# Locate files and folders dynamically
# ----------------------------------------
gps_file = glob.glob(os.path.join(data_dir, "*_CAM_GPS.txt"))
if not gps_file:
    raise FileNotFoundError("No *_CAM_GPS.txt file found in " + data_dir)
gps_log_path = gps_file[0]

click_file = glob.glob(os.path.join(data_dir, "*_Click.csv"))
if not click_file:
    raise FileNotFoundError("No *_Click.csv file found in " + data_dir)
labels_path = click_file[0]

subdirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
rgb_folder = next(
    (os.path.join(data_dir, d) for d in subdirs if d.upper().endswith("_RGB")), None
)
ir_folder = next(
    (os.path.join(data_dir, d) for d in subdirs if d.upper().endswith("_IR")), None
)
if not rgb_folder or not ir_folder:
    raise FileNotFoundError("RGB or IR subdirectory not found in " + data_dir)

# ----------------------------------------
# Load GPS log
# ----------------------------------------
raw_gps = pd.read_csv(gps_log_path, header=None)
gps_df = pd.DataFrame(
    {
        "trigger": raw_gps[0].astype(int),
        "latitude": raw_gps[1].astype(float),
        "longitude": raw_gps[2].astype(float),
        "altitude": raw_gps[3].astype(float),
        "q0": raw_gps[8].astype(float),
        "q1": raw_gps[9].astype(float),
        "q2": raw_gps[10].astype(float),
        "q3": raw_gps[11].astype(float),
    }
)

# ----------------------------------------
# Load click points
# ----------------------------------------
labels_df = pd.read_csv(
    labels_path,
    header=0,
    names=["latitude", "longitude", "unused1", "altitude", "unused2", "label_str"],
)
labels_df["label"] = labels_df["label_str"].str.extract(r"(\d+)").astype(int)


# ----------------------------------------
# Projection helper
# ----------------------------------------
def lonlat_to_webmerc(lon, lat):
    x = lon * 20037508.34 / 180.0
    y = (
        np.log(np.tan((90.0 + lat) * np.pi / 360.0))
        * (20037508.34 / 180.0)
        / (np.pi / 180.0)
    )
    return x, y


merc = np.array(
    [lonlat_to_webmerc(lon, lat) for lon, lat in zip(gps_df.longitude, gps_df.latitude)]
)
merc_x, merc_y = merc[:, 0], merc[:, 1]
gps_alt = gps_df.altitude.values
lbl_merc = np.array(
    [
        lonlat_to_webmerc(lon, lat)
        for lon, lat in zip(labels_df.longitude, labels_df.latitude)
    ]
)
lbl_x, lbl_y, lbl_z = lbl_merc[:, 0], lbl_merc[:, 1], labels_df.altitude.values
lbl_colors = np.where(labels_df.label == 1, "red", "blue")

# Bounding box + padding
all_x = np.concatenate([merc_x, lbl_x])
all_y = np.concatenate([merc_y, lbl_y])
all_z = np.concatenate([gps_alt, lbl_z, [0.0]])
minx, miny, maxx, maxy = all_x.min(), all_y.min(), all_x.max(), all_y.max()
pad_x, pad_y = (maxx - minx) * 0.1, (maxy - miny) * 0.1
minx_b, miny_b, maxx_b, maxy_b = minx - pad_x, miny - pad_y, maxx + pad_x, maxy + pad_y
min_z, max_z = all_z.min(), all_z.max()

# Fetch satellite tile
dexport = "https://services.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/export"
params = {
    "bbox": f"{minx_b},{miny_b},{maxx_b},{maxy_b}",
    "bboxSR": "3857",
    "imageSR": "3857",
    "format": "png",
    "size": "1024,1024",
    "f": "image",
}
resp = requests.get(dexport, params=params)
resp.raise_for_status()
sat_img = Image.open(BytesIO(resp.content)).convert("RGB")
sat_arr = np.array(sat_img)
sat_small = cv2.resize(sat_arr, (256, 256), interpolation=cv2.INTER_LINEAR)
rows, cols = sat_small.shape[:2]
X = np.linspace(minx_b, maxx_b, cols)
Y = np.linspace(maxy_b, miny_b, rows)
X_mesh, Y_mesh = np.meshgrid(X, Y)
sat_rgba = np.zeros((rows, cols, 4), dtype=np.float32)
sat_rgba[..., :3] = sat_small.astype(np.float32) / 255.0
sat_rgba[..., 3] = 0.5


# Quaternions → rotation matrices
def quat_to_mat(q):
    q0, q1, q2, q3 = q
    norm = np.linalg.norm(q)
    q0, q1, q2, q3 = q0 / norm, q1 / norm, q2 / norm, q3 / norm
    return np.array(
        [
            [
                1 - 2 * (q2 * q2 + q3 * q3),
                2 * (q1 * q2 - q3 * q0),
                2 * (q1 * q3 + q2 * q0),
            ],
            [
                2 * (q1 * q2 + q3 * q0),
                1 - 2 * (q1 * q1 + q3 * q3),
                2 * (q2 * q3 - q1 * q0),
            ],
            [
                2 * (q1 * q3 - q2 * q0),
                2 * (q2 * q3 + q1 * q0),
                1 - 2 * (q1 * q1 + q2 * q2),
            ],
        ]
    )


rotated_axes = [
    (quat_to_mat((r.q0, r.q1, r.q2, r.q3)) @ np.eye(3)).T for _, r in gps_df.iterrows()
]

# Preload images + names
rgb_dict, rgb_name = {}, {}
for fn in os.listdir(rgb_folder):
    if fn.lower().endswith((".tiff", ".jpg", ".png")):
        try:
            trig = int(fn.split("#")[-1].split(".")[0])
        except:
            continue
        img = cv2.cvtColor(cv2.imread(os.path.join(rgb_folder, fn)), cv2.COLOR_BGR2RGB)
        rgb_dict[trig] = img
        rgb_name[trig] = fn
ir_dict, ir_name = {}, {}
for fn in os.listdir(ir_folder):
    if "_gray" in fn.lower():
        try:
            trig = int(fn.split("#")[-1].split("_")[0])
        except:
            continue
        img = cv2.cvtColor(cv2.imread(os.path.join(ir_folder, fn)), cv2.COLOR_BGR2RGB)
        ir_dict[trig] = img
        ir_name[trig] = fn
# blank frame dims
h, w = next(iter(rgb_dict.values())).shape[:2]
blank = np.zeros((h, w, 3), dtype=np.uint8)

# Build frame lists
images_rgb = [
    rgb_dict.get(int(r.trigger + rgb_offset), blank) for _, r in gps_df.iterrows()
]
images_ir = [
    ir_dict.get(int(r.trigger + ir_offset), blank) for _, r in gps_df.iterrows()
]


# Helper to rotate an image about center
def rotate_img(img, angle):
    (h, w) = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(img, M, (w, h))


# Figure + axes setup
fig = plt.figure(figsize=(12, 8))
gs = fig.add_gridspec(
    2, 2, width_ratios=[1, 1], height_ratios=[1, 1], hspace=0.2, wspace=0.1
)
ax3d = fig.add_subplot(gs[:, 0], projection="3d")
ax_rgb = fig.add_subplot(gs[0, 1])
ax_ir = fig.add_subplot(gs[1, 1])
ax_rgb.axis("off")
ax_ir.axis("off")
img_plot_rgb = ax_rgb.imshow(blank)
img_plot_ir = ax_ir.imshow(blank)
# 3D static
dx, dy = maxx_b - minx_b, maxy_b - miny_b
dz = max_z - min_z if min_z >= 0 else max_z + abs(min_z)
ax3d.plot_surface(
    X_mesh,
    Y_mesh,
    np.full_like(X_mesh, min_z),
    rcount=rows,
    ccount=cols,
    facecolors=sat_rgba,
    shade=False,
)
ax3d.plot(merc_x, merc_y, gps_alt, color="lightgray", linewidth=2)
ax3d.scatter(merc_x, merc_y, gps_alt, color="yellow", s=5)
ax3d.scatter(lbl_x, lbl_y, lbl_z, c=lbl_colors, s=20)
axis_x, axis_y, axis_z, drone = [
    ax3d.plot([], [], [], color=c, linewidth=2 if c != "blue" else 2)[0]
    for c in ("r", "g", "b")
] + [ax3d.plot([], [], [], marker="o", color="red", markersize=6)[0]]
ax3d.set_xlabel("Web-Mercator X")
ax3d.set_ylabel("Web-Mercator Y")
ax3d.set_zlabel("Altitude (m)")
ax3d.set_box_aspect((dx, dy, dz))


# Animation update
def update_frame(i):
    trig = int(gps_df.iloc[i].trigger)
    # RGB
    rgb_img = rotate_img(images_rgb[i], rgb_angle)
    img_plot_rgb.set_data(rgb_img)
    ax_rgb.set_title(f"RGB: {rgb_name.get(trig, 'blank')}", fontsize=8)
    # IR
    ir_img = rotate_img(images_ir[i], ir_angle)
    img_plot_ir.set_data(ir_img)
    ax_ir.set_title(f"IR: {ir_name.get(trig, 'blank')}", fontsize=8)
    # 3D axes
    o = np.array([merc_x[i], merc_y[i], gps_alt[i]])
    x_e, y_e, z_e = rotated_axes[i]
    L = (dx + dy) / 20
    ends = {"x": o + x_e * L, "y": o + y_e * L, "z": o + z_e * L}
    for ax_line, key in zip((axis_x, axis_y, axis_z), ("x", "y", "z")):
        ax_line.set_data([o[0], ends[key][0]], [o[1], ends[key][1]])
        ax_line.set_3d_properties([o[2], ends[key][2]])
    drone.set_data([merc_x[i]], [merc_y[i]])
    drone.set_3d_properties([gps_alt[i]])
    ax3d.set_title(f"Trigger: {trig}")
    return axis_x, axis_y, axis_z, drone, img_plot_rgb, img_plot_ir


anim = FuncAnimation(fig, update_frame, frames=len(gps_df), interval=200, blit=False)
plt.tight_layout()
plt.show()
