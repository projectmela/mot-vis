from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional, Union, Tuple

import cv2
import numpy as np
from loguru import logger
from tqdm.auto import tqdm

from mot_vis.io import load_mot_annotation


def frame_from_video(video: cv2.VideoCapture, start_frame: int = 0, end_frame: int = -1):
    """ Read frames from video
    :param video: cv2.VideoCapture object
    :param start_frame: start frame number to read
    :param end_frame: end frame number to read
    :return: generator of frames
    """
    num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    end_frame = end_frame if end_frame > 0 else num_frames - 1
    assert start_frame >= 0, f"start_frame ({start_frame}) must be >= 0"
    assert start_frame < num_frames, f"start_frame ({start_frame}) must be < num_frames ({num_frames})"
    assert end_frame > start_frame, f"end_frame ({end_frame}) must be > start_frame ({start_frame})"
    assert end_frame < num_frames, f"end_frame ({end_frame}) must be < num_frames ({num_frames})"

    # set to start frame
    video.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    frame_count = start_frame
    while video.isOpened():
        if frame_count > end_frame:
            logger.debug(f"reached end frame {end_frame}.")
            break
        success, frame = video.read()
        if success:
            frame_count += 1
            yield frame
        else:
            logger.debug(f"video ended after frame {frame_count}.")
            break


def plot_mot_bbox_on_image(
        img: np.ndarray,
        mot_dets: np.ndarray,
        color_map: List[Tuple[int]] = None,
        line_width: int = 2,
        font_scale=1.3,
        font_thickness=3,
) -> np.ndarray:
    """plot MOT format detections on a frame
    :param img: frame to plot on, read from openCV
    :param mot_dets: MOT format detections of the image, shape (n_dets, 10)
    MOT format: https://motchallenge.net/instructions/
    <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <x>, <y>, <z>
    :param color_map: color map for each class, shape (n_classes, 3), default to None; will un-normalize if needed
    :param line_width: line width of the bounding box
    :param font_scale: font scale of the id
    :param font_thickness: font thickness of the id
    :return: frame with detections plotted
    """
    assert img.ndim == 3, "img should be a 3D array"
    assert mot_dets.ndim == 2, "mot annotations should be a 2D array"
    assert mot_dets.shape[1] == 10, "mot annotations should have 10 columns"

    if color_map is None:
        # create random colors for each object
        n_objects = 200
        np.random.seed(42)
        colors = np.random.rand(n_objects, 3)
        # convert to tuple(int) and de-normalize
        color_map = [tuple(map(int, c * 255)) for c in colors]

    # convert to [x1,y1,w,h] to [x1,y1,x2,y2]
    mot_dets[:, 4:6] += mot_dets[:, 2:4]
    for d in mot_dets:
        obj_id = int(d[1])
        color = color_map[obj_id % len(color_map)]
        x1, y1, x2, y2 = d[2:6].astype(int)

        # plot bounding box
        cv2.rectangle(img, (x1, y1), (x2, y2), color, line_width)

        # plot id
        shift = 8
        cv2.putText(img, str(obj_id), (x1 - shift, y1 - shift),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, font_thickness)

    return img


def plot_frames_to_frames(
        frame_paths: List[Union[str, Path]],
        mot_dets: np.ndarray,
        save_dir_path: Union[str, Path],
):
    """plot MOT format predictions (annotations) on frames from frame_paths and save it as frames
    :param frame_paths: list of paths to frames
    :param mot_dets: MOT format detections, shape (n_dets, 10), MOT format: https://motchallenge.net/instructions/
    <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <x>, <y>, <z>
    :param save_dir_path: path to save frames with detections plotted
    """

    assert mot_dets.ndim == 2, "mot dets should be a 2D array"
    assert mot_dets.shape[1] == 10, "mot dets should have 10 columns"

    save_dir_path = Path(save_dir_path) / "plotted"
    save_dir_path.mkdir(parents=True, exist_ok=True)

    def _process_frame(f_path, i, _mot_dets):
        f_path = Path(f_path)
        img = cv2.imread(f_path.as_posix())
        plotted_img = plot_mot_bbox_on_image(img, _mot_dets[
            _mot_dets[:, 0] == i])  # BUG FIX : Indexing error earlier used i + 1 used to plot wrong boxes
        img_save_path = save_dir_path / f_path.name
        cv2.imwrite(img_save_path.as_posix(), plotted_img)
        return f_path.name

    # x1.3 faster than tqdm.thread_map; x5.3 faster than for-loop
    # TODO: which OpenCV function does not support multi-processing?
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(_process_frame, f_path, i, mot_dets) for i, f_path in enumerate(frame_paths)]
        # noinspection PyTypeChecker
        for f in tqdm(as_completed(futures), total=len(futures), desc="plotting frames"):
            _ = f.result()


def plot_from_video_to_video(
        src_video_path: Union[Path, str],
        mot_anno_path: Union[Path, str],
        frame_rate: float = None,
        save_path: Optional[Union[str, Path]] = None,
):
    """plot MOT format predictions (annotations) on frames read from a video and save it to a new video
    :param src_video_path: path to the video to plot on
    :param mot_anno_path: path to the MOT format annotation file
    [MOT format](https://motchallenge.net/instructions/):
    <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <x>, <y>, <z>
    :param frame_rate: frame rate of the output video
    :param save_path: file path or dir path to save the output video, default to the directory of the video file
    """
    src_video_path = Path(src_video_path).absolute()
    save_path = Path(save_path) if save_path else mot_anno_path.parent
    mot_anno = load_mot_annotation(mot_anno_path)

    assert src_video_path.exists(), f"video file not found: {src_video_path}"

    vc = cv2.VideoCapture(src_video_path.as_posix())
    assert vc.isOpened(), f"Failed to open video at {src_video_path}"
    video_name = src_video_path.name

    # video specs for writing video
    width = int(vc.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vc.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames_per_second = frame_rate or vc.get(cv2.CAP_PROP_FPS)
    num_frames = int(vc.get(cv2.CAP_PROP_FRAME_COUNT))
    codec, file_ext = ("mp4v", ".mp4")

    # prepare directory to save video and the path to the plotted video
    if save_path.is_dir():
        output_dir_path = save_path
        # path to the plotted video file
        output_v_path = output_dir_path / f"{video_name.split('.')[0]}_plotted{file_ext}"
    else:
        output_v_path = save_path
        output_dir_path = output_v_path.parent
        output_dir_path.mkdir(parents=True, exist_ok=True)
    assert not output_v_path.exists(), f"output file already exists: {output_v_path}"

    vw = cv2.VideoWriter(
        filename=output_v_path.as_posix(),
        # some installation of opencv may not support x264 (due to its license),
        fourcc=cv2.VideoWriter_fourcc(*codec),
        fps=float(frames_per_second),
        frameSize=(width, height),
        isColor=True,
    )

    logger.info(
        "Reading video from:\n"
        f"{src_video_path}\n"
        f"{width}x{height} @ {frames_per_second:.2f} fps with {num_frames} frames\n"
        f"Writing video to:\n{output_v_path}"
    )

    num_frames = int(vc.get(cv2.CAP_PROP_FRAME_COUNT))
    logger.info(f"Plotting {num_frames} frames ...")
    frame_gen = frame_from_video(vc)
    # noinspection PyTypeChecker
    for frame_id, img in enumerate(tqdm(frame_gen, total=num_frames, desc="plotting frames")):
        frame_id += 1  # MOT format starts from 1, but frame_id from video capture starts from 0
        dets = mot_anno[mot_anno[:, 0] == frame_id]
        plotted_frame = plot_mot_bbox_on_image(img=img, mot_dets=dets)
        vw.write(plotted_frame)

    vw.release()
    vc.release()


def plot_frames_to_video(
        frame_paths: List[Union[str, Path]],
        mot_anno_path: Union[Path, str],
        save_path: Optional[Union[str, Path]] = None,
        frame_rate: float = 30.0,
):
    """plot MOT format predictions (annotations) on frames from frame_paths and save it as a new video
    :param frame_paths: list of paths to frames
    :param mot_dets: MOT format detections, shape (n_dets, 10), MOT format: https://motchallenge.net/instructions/
    <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <x>, <y>, <z>
    :param frame_rate: frame rate of the output video
    :param save_path: file path or dir path to save the output video, default to the directory of the frame paths
    """

    mot_anno = load_mot_annotation(mot_anno_path)

    if isinstance(frame_paths[0], str):
        frame_paths = [Path(f) for f in frame_paths]

    newline = "\n"
    logger.debug(
        "Video frames will be in the order of the frame paths.\n"
        "Here is the order of first 5 images:\n"
        f"{newline.join([str(p) for p in frame_paths[:5]])}"
    )
    save_path = Path(save_path) if save_path is not None else frame_paths[0].parent

    # video specs for writing video
    eg_frame = cv2.imread(frame_paths[0].as_posix())
    h, w, _ = eg_frame.shape
    del eg_frame
    codec, file_ext = ("mp4v", ".mp4")

    # video name is the folder name of the frames
    video_name = save_path.name

    # prepare directory to save video and the path to the plotted video
    if save_path.is_dir():
        output_dir_path = save_path
        # path to the plotted video file
        output_v_path = output_dir_path / f"{video_name}_plotted{file_ext}"
    else:
        output_v_path = save_path
        output_dir_path = output_v_path.parent
        output_dir_path.mkdir(parents=True, exist_ok=True)
    assert not output_v_path.exists(), f"output file already exists: {output_v_path}"

    # create video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(output_v_path.as_posix(), fourcc, frame_rate, (w, h))
    # noinspection PyTypeChecker
    for i, frame_path in enumerate(tqdm(frame_paths, desc="Plotting frames")):
        frame_id = i
        img = cv2.imread(frame_path.as_posix())
        dets = mot_anno[mot_anno[:, 0] == frame_id]
        plotted_frame = plot_mot_bbox_on_image(img=img, mot_dets=dets)
        vw.write(plotted_frame)

    vw.release()
    cv2.destroyAllWindows()
