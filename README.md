# MOT Visualization

Utilities to visualize Multi-object tracking (MOT) results.

## Tracking Annotation / Prediction Format

Accept MOTChallenge MOT17/20 format in `.txt` file, see [here](https://motchallenge.net/instructions/).

```text
frame_id, id, bb_left, bb_top, bb_width, bb_height, conf, x, y, z
```

Frames starting from 1 is strictly required.

## Usage

- Annotation: a single `.txt`` file with the format above.
- Media: Image directory / video

```angular2html
# plot from video/frames to video
python plot_to_video.py \
    -s path/to/video.mp4 \
    -a path/to/mot.txt

python plot_to_video.py \
    -s path/to/frame_file_dir \
    -a path/to/mot.txt
```
