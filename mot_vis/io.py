import re
from pathlib import Path
from typing import Union

import numpy as np
from loguru import logger


def load_mot_annotation(fp: Union[str, Path]) -> np.ndarray:
    """Load MOT annotations from a .txt file.
    :param fp: File path to a MOT format .txt file.
    :return: MOT annotations in np.ndarray.
    """
    fp = Path(fp)
    assert fp.exists(), f"File does not exist at {fp}"
    if fp.suffix == ".txt":
        dets = np.loadtxt(fp.as_posix(), delimiter=",", dtype=np.float32)
    else:
        raise ValueError(f"MOT format .txt file is required, got: file {fp.suffix}.")
    verify_mot_format(dets)
    return dets


def verify_mot_format(mot_dets: np.ndarray) -> None:
    """verify mot format annotations,
    :param mot_dets: mot format annotations
    """
    # frame id should start with 1 (1-based)
    assert mot_dets[:, 0].min() == 1, "Frame id should start with 1 (MOT format is 1-based)"
    # check no nan
    assert not np.isnan(mot_dets).any(), "NaN detected in MOT format annotations"
    # check shape
    assert mot_dets.shape[1] == 10, "MOT format annotations should have 10 columns"


def sort_frame_files(frame_files: list[Union[str, Path]]) -> list[Union[str, Path]]:
    """sort frame files by frame number
    :param frame_files: list of frame file paths
    :return: sorted frame file paths
    """
    frame_files = sorted(frame_files, key=get_frame_number_from_path)
    return frame_files


def get_frame_number_from_path(frame_path: Union[str, Path]) -> int:
    """get frame number from frame path, the last number section in the file name
    :param frame_path: path to a frame file
    :return: frame number
    """
    frame_path = Path(frame_path)
    return int(re.findall(r"\d+", frame_path.stem)[-1])


def get_frame_paths(frames_dir: Union[Path, str]) -> list[Path]:
    """get frame paths from a directory
    :param frames_dir: path to a directory containing frames
    :return: list of frame paths
    """
    frames_dir = Path(frames_dir)
    frame_paths = []

    img_exts = ["jpg", "JPG", "jpeg", "JPEG", "png", "PNG"]

    for ext in img_exts:
        frame_paths = list(frames_dir.glob(f"*.{ext}"))
        if frame_paths:
            frame_paths = sort_frame_files(frame_paths)
            break

    newline = "\n"
    logger.warning(
        "Order of frame paths read as following:\n"
        f"{newline.join([str(p) for p in frame_paths[:5]])}"
    )

    return frame_paths
