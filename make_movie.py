# make_movie.py

import cv2
import os
import glob

def make_hd_video(image_folder, output_path, fps=10):
    width, height = 1920, 1080  # HD resolution
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    image_files = sorted(glob.glob(os.path.join(image_folder, 'tracking*.png')))

    if not image_files:
        raise FileNotFoundError(f"No images found in {image_folder} with pattern 'tracking*.png'.")

    for img_path in image_files:
        frame = cv2.imread(img_path)
        if frame is None:
            print(f"⚠️ Skipping unreadable frame: {img_path}")
            continue
        frame = cv2.resize(frame, (width, height))  # force HD resize
        video_out.write(frame)

    video_out.release()
    print(f"✅ Video saved to {output_path}")

if __name__ == "__main__":
    make_hd_video(image_folder='output', output_path='tracking_output.mp4', fps=10)
