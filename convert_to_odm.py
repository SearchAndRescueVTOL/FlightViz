import os
import glob
import argparse
import pandas as pd
import piexif
from PIL import Image
import numpy as np


def dms_to_rational(degree_float):
    degrees = int(degree_float)
    minutes_full = (degree_float - degrees) * 60
    minutes = int(minutes_full)
    seconds = round((minutes_full - minutes) * 60, 6)
    return [(degrees, 1), (minutes, 1), (int(seconds * 1000000), 1000000)]


def quaternion_to_yaw(q0, q1, q2, q3):
    # Compute yaw (heading) in degrees from quaternion (ENU convention)
    # yaw = atan2(2*(q0*q3 + q1*q2), 1 - 2*(q2*q2 + q3*q3))
    yaw_rad = np.arctan2(2 * (q0 * q3 + q1 * q2), 1 - 2 * (q2 * q2 + q3 * q3))
    yaw_deg = np.degrees(yaw_rad) % 360
    return yaw_deg


def embed_gps_exif(img_path, lat, lon, alt, yaw, out_path):
    img = Image.open(img_path)
    exif_bytes = img.info.get("exif")
    if exif_bytes:
        exif_dict = piexif.load(exif_bytes)
    else:
        exif_dict = {"0th": {}, "Exif": {}, "GPS": {}, "1st": {}, "thumbnail": None}

    gps_ifd = {}
    gps_ifd[piexif.GPSIFD.GPSLatitudeRef] = "N" if lat >= 0 else "S"
    gps_ifd[piexif.GPSIFD.GPSLatitude] = dms_to_rational(abs(lat))
    gps_ifd[piexif.GPSIFD.GPSLongitudeRef] = "E" if lon >= 0 else "W"
    gps_ifd[piexif.GPSIFD.GPSLongitude] = dms_to_rational(abs(lon))
    gps_ifd[piexif.GPSIFD.GPSAltitudeRef] = 0 if alt >= 0 else 1
    gps_ifd[piexif.GPSIFD.GPSAltitude] = (int(abs(alt) * 100), 100)
    # Embed heading
    gps_ifd[piexif.GPSIFD.GPSImgDirectionRef] = "T"
    gps_ifd[piexif.GPSIFD.GPSImgDirection] = (int(yaw * 100), 100)

    exif_dict["GPS"] = gps_ifd
    out_exif = piexif.dump(exif_dict)
    img.save(out_path, exif=out_exif)


def process_folder(img_folder, gps_df, offset, rotate_angle, out_subdir):
    """
    Process either RGB or IR images in img_folder, geotagging them into out_subdir.
    Uses offset for trigger and rotate_angle for image rotation.
    """
    os.makedirs(out_subdir, exist_ok=True)
    # Determine if this is IR (thermal) folder by out_subdir name
    is_ir = "thermal" in out_subdir.lower()

    blank_count = 0
    processed_count = 0
    for fn in sorted(os.listdir(img_folder)):
        if not fn.lower().endswith((".tif", ".tiff", ".jpg", ".jpeg", ".png")):
            continue
        base = os.path.basename(fn)
        # Extract trigger differently for RGB vs IR
        try:
            if is_ir:
                # filenames like '*_#<trigger>_gray.tiff'
                trigger_str = base.split("#")[-1].split("_")[0]
            else:
                # RGB filenames like '*#<trigger>.<ext>'
                trigger_str = base.split("#")[-1].split(".")[0]
            trigger = int(trigger_str) + offset
        except ValueError:
            print(f"Warning: couldn't parse trigger from {fn}, skipping")
            blank_count += 1
            continue
        if trigger not in gps_df.index:
            print(f"Warning: trigger {trigger} not in GPS log, skipping {fn}")
            blank_count += 1
            continue
        # Retrieve GPS & yaw
        row = gps_df.loc[trigger]
        lat, lon, alt, yaw = row["lat"], row["lon"], row["alt"], row["yaw"]
        src = os.path.join(img_folder, fn)
        name, ext = os.path.splitext(fn)
        # Convert TIFFs to JPEG for compatibility
        if ext.lower() in (".tif", ".tiff"):
            base_name = name
        else:
            base_name = os.path.splitext(fn)[0]
        # Append suffix for ODM recognition
        if is_ir:
            dst_name = f"{base_name}_T.jpg"
        else:
            dst_name = f"{base_name}_RGB.jpg"
        dst = os.path.join(out_subdir, dst_name)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        try:
            embed_gps_exif(src, lat, lon, alt, yaw, dst)
            print(f"Saved geotagged image: {dst}")
            processed_count += 1
        except Exception as e:
            print(f"Failed to geotag {src}: {e}")
    return processed_count, blank_count


def main():
    parser = argparse.ArgumentParser(
        description="Convert dataset to ODM-ready RGB and thermal images with heading"
    )
    parser.add_argument("data_dir", help="Directory containing flight data")
    parser.add_argument("output_dir", help="Directory to write ODM project")
    parser.add_argument(
        "--rgb-offset", type=int, default=0, help="Trigger offset for RGB"
    )
    parser.add_argument(
        "--ir-offset", type=int, default=0, help="Trigger offset for IR"
    )
    parser.add_argument(
        "--rgb-rotate", type=float, default=0.0, help="Rotate RGB images"
    )
    parser.add_argument("--ir-rotate", type=float, default=0.0, help="Rotate IR images")
    args = parser.parse_args()

    data_dir = args.data_dir
    out_dir = args.output_dir
    rgb_offset, ir_offset = args.rgb_offset, args.ir_offset
    rgb_angle, ir_angle = args.rgb_rotate, args.ir_rotate

    os.makedirs(out_dir, exist_ok=True)

    gps_files = glob.glob(os.path.join(data_dir, "*_CAM_GPS.txt"))
    if not gps_files:
        raise FileNotFoundError("No GPS log found in " + data_dir)
    raw = pd.read_csv(gps_files[0], header=None)
    gps_df = pd.DataFrame(
        {
            "trigger": raw[0].astype(int),
            "lat": raw[1].astype(float),
            "lon": raw[2].astype(float),
            "alt": raw[3].astype(float),
            "q0": raw[8].astype(float),
            "q1": raw[9].astype(float),
            "q2": raw[10].astype(float),
            "q3": raw[11].astype(float),
        }
    ).set_index("trigger")
    # compute yaw
    gps_df["yaw"] = gps_df.apply(
        lambda r: quaternion_to_yaw(r.q0, r.q1, r.q2, r.q3), axis=1
    )

    subdirs = [
        d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))
    ]
    # find folders by case-insensitive suffix
    rgb_folder = next((d for d in subdirs if d.lower().endswith("_rgb")), None)
    ir_folder = next((d for d in subdirs if d.lower().endswith("_ir")), None)
    if not rgb_folder:
        print("Available subdirectories:", subdirs)
        raise FileNotFoundError("No RGB folder detected (looking for suffix '_rgb')")
    if not ir_folder:
        print("Available subdirectories:", subdirs)
        raise FileNotFoundError("No IR folder detected (looking for suffix '_ir')")
    if not rgb_folder or not ir_folder:
        raise FileNotFoundError("RGB or IR folder missing")

    rgb_path = os.path.join(data_dir, rgb_folder)
    ir_path = os.path.join(data_dir, ir_folder)

    rgb_out = os.path.join(out_dir, "images")
    n_rgb, b_rgb = process_folder(rgb_path, gps_df, rgb_offset, rgb_angle, rgb_out)
    ir_out = os.path.join(out_dir, "images-thermal")
    n_ir, b_ir = process_folder(ir_path, gps_df, ir_offset, ir_angle, ir_out)

    print(f"Completed: {n_rgb} RGB, {n_ir} IR images geotagged with heading.")
    if b_rgb or b_ir:
        print(f"Skipped {b_rgb} RGB and {b_ir} IR due to missing triggers.")


if __name__ == "__main__":
    main()
