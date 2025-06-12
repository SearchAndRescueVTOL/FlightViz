import os
import glob
import argparse
import pandas as pd
import numpy as np
import requests
from PIL import Image
from io import BytesIO
import cv2

from mayavi import mlab
import pylab as pl
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.animation import FuncAnimation


def mlab_imshowColor(im, alpha=255, **kwargs):
    """
    Plot a color image with mayavi.mlab.imshow.
    - im: ndarray of shape (n, m, 3), values in [0..255]
    - alpha: scalar or 1D array with length n*m, values in [0..255]
    - kwargs passed to mlab.imshow
    """
    try:
        alpha_flat = alpha.flatten().astype(np.uint8)
    except:
        alpha_flat = (pl.ones(im.shape[0] * im.shape[1]) * alpha).astype(np.uint8)

    rgba = np.c_[im.reshape(-1, 3), alpha_flat]
    lut_lookup = pl.arange(im.shape[0] * im.shape[1]).reshape(im.shape[0], im.shape[1])

    img_obj = mlab.imshow(lut_lookup, colormap="binary", **kwargs)
    img_obj.module_manager.scalar_lut_manager.lut.table = rgba
    mlab.draw()
    return img_obj


def lonlat_to_webmerc(lon, lat):
    x = lon * 20037508.34 / 180.0
    y = (
        np.log(np.tan((90.0 + lat) * np.pi / 360.0))
        * (20037508.34 / 180.0)
        / (np.pi / 180.0)
    )
    return x, y


def quaternion_to_rotation_matrix(q):
    q0, q1, q2, q3 = q
    norm = np.sqrt(q0 * q0 + q1 * q1 + q2 * q2 + q3 * q3)
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


def rotate_img(im, angle):
    h, w = im.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(im, M, (w, h))


def main():
    parser = argparse.ArgumentParser(
        description="Visualize drone flight with Mayavi + Matplotlib"
    )
    parser.add_argument("data_dir", help="Directory containing flight data")
    parser.add_argument("--rgb-offset", type=int, default=0, help="RGB trigger offset")
    parser.add_argument("--ir-offset", type=int, default=0, help="IR trigger offset")
    parser.add_argument(
        "--rgb-rotate", type=float, default=0.0, help="Degrees to rotate RGB frames"
    )
    parser.add_argument(
        "--ir-rotate", type=float, default=0.0, help="Degrees to rotate IR frames"
    )
    args = parser.parse_args()

    data_dir = args.data_dir
    rgb_offset = args.rgb_offset
    ir_offset = args.ir_offset
    rgb_angle = args.rgb_rotate
    ir_angle = args.ir_rotate

    # Extract date prefix from folder name
    folder_name = os.path.basename(os.path.normpath(data_dir))
    parts = folder_name.split("_")
    date = "_".join(parts[0:2]) if len(parts) >= 2 else parts[0]

    # Locate click CSV by date
    click_csv = os.path.join(data_dir, f"{date}_Click.csv")
    if not os.path.isfile(click_csv):
        raise FileNotFoundError(f"Click CSV not found: {click_csv}")

    # Locate GPS log by date pattern
    gps_candidates = glob.glob(os.path.join(data_dir, f"{date}*_CAM_GPS.txt"))
    if not gps_candidates:
        raise FileNotFoundError(f"GPS log not found for date {date}")
    gps_log = gps_candidates[0]

    # Locate RGB and IR folders by date pattern
    rgb_dirs = glob.glob(os.path.join(data_dir, f"{date}*_RGB"))
    ir_dirs = glob.glob(os.path.join(data_dir, f"{date}*_IR"))
    if not rgb_dirs or not ir_dirs:
        raise FileNotFoundError("RGB or IR subdirectory not found using date prefix")
    rgb_folder = rgb_dirs[0]
    ir_folder = ir_dirs[0]

    # Load GPS log
    raw = pd.read_csv(gps_log, header=None)
    gps_df = pd.DataFrame(
        {
            "trigger": raw[0].astype(int),
            "latitude": raw[1].astype(float),
            "longitude": raw[2].astype(float),
            "altitude": raw[3].astype(float),
            "q0": raw[8].astype(float),
            "q1": raw[9].astype(float),
            "q2": raw[10].astype(float),
            "q3": raw[11].astype(float),
        }
    )

    # Load click points
    labels = pd.read_csv(
        click_csv,
        header=0,
        names=["latitude", "longitude", "u1", "altitude", "u2", "label_str"],
    )
    labels["label"] = labels["label_str"].str.extract(r"(\d+)").astype(int)

    # Project coordinates to Web Mercator
    merc = np.array(
        [
            lonlat_to_webmerc(lon, lat)
            for lon, lat in zip(gps_df.longitude, gps_df.latitude)
        ]
    )
    mx, my = merc[:, 0], merc[:, 1]
    alt = gps_df.altitude.values
    lbl_m = np.array(
        [
            lonlat_to_webmerc(lon, lat)
            for lon, lat in zip(labels.longitude, labels.latitude)
        ]
    )
    lx, ly = lbl_m[:, 0], lbl_m[:, 1]
    lz = labels.altitude.values

    # Compute bounding square
    all_x = np.concatenate((mx, lx))
    all_y = np.concatenate((my, ly))
    all_z = np.concatenate((alt, lz, [0]))
    minx, maxx = all_x.min(), all_x.max()
    miny, maxy = all_y.min(), all_y.max()
    minz, maxz = all_z.min(), all_z.max()
    dx, dy = maxx - minx, maxy - miny
    L = max(dx, dy)
    cx, cy = (minx + maxx) / 2, (miny + maxy) / 2
    minx_s, maxx_s = cx - L / 2, cx + L / 2
    miny_s, maxy_s = cy - L / 2, cy + L / 2

    # Fetch color satellite tile
    url = (
        "https://services.arcgisonline.com/ArcGIS/rest/services/"
        "World_Imagery/MapServer/export"
    )
    params = {
        "bbox": f"{minx_s},{miny_s},{maxx_s},{maxy_s}",
        "bboxSR": "3857",
        "imageSR": "3857",
        "format": "png",
        "size": "1024,1024",
        "f": "image",
    }
    resp = requests.get(url, params=params)
    resp.raise_for_status()
    sat_img = Image.open(BytesIO(resp.content)).convert("RGB")
    sat_small = cv2.resize(
        np.array(sat_img), (256, 256), interpolation=cv2.INTER_LINEAR
    )
    sat_small = sat_small.transpose((1, 0, 2))  # transpose so axes align correctly

    # Precompute body axes
    rotated_axes = []
    for _, row in gps_df.iterrows():
        R = quaternion_to_rotation_matrix((row.q0, row.q1, row.q2, row.q3))
        x_ned = R @ np.array([1, 0, 0])
        y_ned = R @ np.array([0, 1, 0])
        z_ned = R @ np.array([0, 0, 1])
        x_enu = np.array([x_ned[1], x_ned[0], -x_ned[2]])
        y_enu = np.array([y_ned[1], y_ned[0], -y_ned[2]])
        z_enu = np.array([z_ned[1], z_ned[0], -z_ned[2]])
        rotated_axes.append((x_enu, y_enu, z_enu))

    # Load and rotate RGB images
    img_files = sorted(glob.glob(os.path.join(rgb_folder, f"{date}*#*.*")))
    rgb_imgs, rgb_names = [], []
    for fn in img_files:
        try:
            t = int(os.path.basename(fn).split("#")[-1].split(".")[0])
        except:
            continue
        img = cv2.cvtColor(cv2.imread(fn), cv2.COLOR_BGR2RGB)
        rgb_imgs.append((t, rotate_img(img, rgb_angle)))
        rgb_names.append((t, os.path.basename(fn)))
    rgb_imgs.sort(key=lambda x: x[0])
    rgb_names.sort(key=lambda x: x[0])
    images_rgb = [img for _, img in rgb_imgs]
    names_rgb = {t: name for t, name in rgb_names}

    # Load and rotate IR images
    ir_files = sorted(glob.glob(os.path.join(ir_folder, f"{date}*#*gray.*")))
    ir_imgs, ir_names = [], []
    for fn in ir_files:
        try:
            t = int(os.path.basename(fn).split("#")[-1].split("_")[0])
        except:
            continue
        img = cv2.cvtColor(cv2.imread(fn), cv2.COLOR_BGR2RGB)
        ir_imgs.append((t, rotate_img(img, ir_angle)))
        ir_names.append((t, os.path.basename(fn)))
    ir_imgs.sort(key=lambda x: x[0])
    ir_names.sort(key=lambda x: x[0])
    images_ir = [img for _, img in ir_imgs]
    names_ir = {t: name for t, name in ir_names}

    # Mayavi 3D scene
    mlab.figure("Flight Viz", size=(1200, 800), bgcolor=(0, 0, 0))
    mlab_imshowColor(
        sat_small,
        alpha=128,
        extent=[minx_s, maxx_s, miny_s, maxy_s, minz - 1, minz - 1],
    )
    mlab.plot3d(mx, my, alt, color=(0.8, 0.8, 0.8), tube_radius=None, line_width=2)
    mlab.points3d(mx, my, alt, color=(1, 1, 0), scale_factor=L * 0.005)
    mlab.points3d(lx, ly, lz, color=(1, 0, 0), scale_factor=L * 0.01)
    axis_length = L / 20
    ax_x = mlab.plot3d([0, 0], [0, 0], [0, 0], color=(1, 0, 0), tube_radius=L * 0.001)
    ax_y = mlab.plot3d([0, 0], [0, 0], [0, 0], color=(0, 1, 0), tube_radius=L * 0.001)
    ax_z = mlab.plot3d([0, 0], [0, 0], [0, 0], color=(0, 0, 1), tube_radius=L * 0.001)
    drone = mlab.points3d(mx[0], my[0], alt[0], color=(1, 0, 0), scale_factor=L * 0.01)
    txt = mlab.text(0.02, 0.03, f"Trigger {gps_df.trigger.iloc[0]}", width=0.06)
    mlab.view(azimuth=0, elevation=90, distance="auto")

    # Matplotlib panel
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 4))
    plt.subplots_adjust(bottom=0.25)
    im1 = ax1.imshow(images_rgb[0])
    ax1.axis("off")
    im2 = ax2.imshow(images_ir[0])
    ax2.axis("off")
    ax1.set_title(f"RGB: {names_rgb.get(gps_df.trigger.iloc[0], '')}")
    ax2.set_title(f"IR: {names_ir.get(gps_df.trigger.iloc[0], '')}")
    slider_ax = plt.axes([0.15, 0.1, 0.7, 0.03])
    slider = Slider(slider_ax, "Delay", 10, 1000, valinit=200, valstep=1)

    def update(i):
        idx = i % len(rotated_axes)
        t = gps_df.trigger.iloc[idx]
        im1.set_data(images_rgb[idx])
        ax1.set_title(f"RGB: {names_rgb.get(t, '')} (Trig {t})")
        im2.set_data(images_ir[idx])
        ax2.set_title(f"IR: {names_ir.get(t, '')} (Trig {t})")
        o = np.array([mx[idx], my[idx], alt[idx]])
        xe, ye, ze = rotated_axes[idx]
        ends = {
            "x": o + xe * axis_length,
            "y": o + ye * axis_length,
            "z": o + ze * axis_length,
        }
        ax_x.mlab_source.set(
            x=[o[0], ends["x"][0]], y=[o[1], ends["x"][1]], z=[o[2], ends["x"][2]]
        )
        ax_y.mlab_source.set(
            x=[o[0], ends["y"][0]], y=[o[1], ends["y"][1]], z=[o[2], ends["y"][2]]
        )
        ax_z.mlab_source.set(
            x=[o[0], ends["z"][0]], y=[o[1], ends["z"][1]], z=[o[2], ends["z"][2]]
        )
        drone.mlab_source.set(x=[mx[idx]], y=[my[idx]], z=[alt[idx]])
        txt.text = f"Trigger {t}"
        mlab.draw()
        return im1, im2

    anim = FuncAnimation(
        fig, update, frames=len(rotated_axes), interval=200, blit=False
    )
    slider.on_changed(lambda v: setattr(anim.event_source, "interval", int(v)))
    plt.show()
    mlab.show()


if __name__ == "__main__":
    main()
