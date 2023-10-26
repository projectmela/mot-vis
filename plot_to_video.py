import argparse
from pathlib import Path

from loguru import logger

from mot_viz import plot_frames_to_video, plot_from_video_to_video
from mot_viz.io import get_frame_paths

parser = argparse.ArgumentParser(description="Plotting detection on the image.")
parser.add_argument("-s", "--src", type=str, required=True,
                    help="Path of the video or frames directory")
parser.add_argument("-a", "--annotation", type=str, required=True,
                    help="Path of the annotation file")
parser.add_argument("-o", "--output", type=str, required=False,
                    help="Path of where to save the output video file; "
                         "default to be the same directory as the annotation file")

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)

    src_path = Path(args.src)
    anno_file_path = Path(args.annotation)
    output_dir_path = args.output or anno_file_path.parent

    assert src_path.exists(), f"File does not exist at {src_path}"
    assert anno_file_path.exists(), f"File does not exist at {anno_file_path}"
    assert anno_file_path.suffix == ".txt", f"Annotation file should be a .txt file, got {anno_file_path.suffix}"

    if src_path.is_dir():
        logger.info(f"Scanning images in {src_path.as_posix()}")
        frame_paths = get_frame_paths(src_path)
        plot_frames_to_video(frame_paths, anno_file_path, output_dir_path)
    else:
        video_exts = [".mp4", ".avi", ".mov"]
        video_exts += [ext.upper() for ext in video_exts] # add uppercase extensions
        assert src_path.suffix in video_exts, f"Video file should have extension in {video_exts}, got {src_path.suffix}"
        plot_from_video_to_video(src_video_path=src_path, mot_anno_path=anno_file_path, save_path=output_dir_path)
