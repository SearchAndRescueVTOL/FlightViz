import os
import glob
import argparse
import numpy as np
import cv2


def process_folder(
    input_dir: str, output_dir: str, width: int = 640, height: int = 512
):
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    frame_size = width * height * 3 // 2  # YUV420p has 1.5 bytes per pixel

    pattern = os.path.join(input_dir, "*.raw")
    for raw_path in glob.glob(pattern):
        with open(raw_path, "rb") as f:
            raw = f.read()

        if len(raw) != frame_size:
            print(
                f"Skipping {os.path.basename(raw_path)}: wrong size ({len(raw)} bytes)"
            )
            continue

        # Extract Y plane only (grayscale)
        y_plane = np.frombuffer(raw, dtype=np.uint8, count=width * height)
        y_image = y_plane.reshape((height, width))

        # Build output filename and save
        base = os.path.splitext(os.path.basename(raw_path))[0]
        out_path = os.path.join(output_dir, f"{base}_gray.tiff")
        cv2.imwrite(out_path, y_image)
        print(f"Saved {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert YUV420p .raw frames to grayscale TIFF images."
    )
    parser.add_argument("input_dir", help="Path to the folder containing .raw files")
    parser.add_argument(
        "output_dir", help="Path where converted .tiff files will be saved"
    )
    parser.add_argument(
        "--width", type=int, default=640, help="Frame width in pixels (default: 640)"
    )
    parser.add_argument(
        "--height", type=int, default=512, help="Frame height in pixels (default: 512)"
    )
    args = parser.parse_args()

    process_folder(args.input_dir, args.output_dir, args.width, args.height)


if __name__ == "__main__":
    main()
