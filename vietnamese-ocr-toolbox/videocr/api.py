import argparse

from .video import Video
from tool.config import Config


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="./results", help="Path to output folder")
    parser.add_argument(
        "--debug", action="store_true", help="Save every steps for debugging"
    )
    parser.add_argument(
        "--do_retrieve", action="store_true", help="Whether to retrive information"
    )
    parser.add_argument(
        "--find_best_rotation",
        action="store_true",
        help="Whether to find rotation of document in the image",
    )
    args = parser.parse_args()

    return args


def get_subtitles(
    video_path: str,
    time_start="0:00",
    time_end="",
    conf_threshold=75,
    sim_threshold=80,
    use_fullframe=False,
    brightness_threshold=None,
    similar_image_threshold=100,
    similar_pixel_threshold=25,
    frames_to_skip=1,
    crop_x=None,
    crop_y=None,
    crop_width=None,
    crop_height=None,
) -> str:

    args = get_args()
    config = Config("/app/vietnamese-ocr-toolbox/tool/config/configs.yaml")

    v = Video(video_path)
    v.run_ocr(
        args,
        config,
        time_start,
        time_end,
        conf_threshold,
        use_fullframe,
        brightness_threshold,
        similar_image_threshold,
        similar_pixel_threshold,
        frames_to_skip,
        crop_x,
        crop_y,
        crop_width,
        crop_height,
    )
    return v.get_subtitles(sim_threshold)


def save_subtitles_to_file(
    video_path: str,
    file_path="subtitle.srt",
    time_start="0:00",
    time_end="",
    conf_threshold=75,
    sim_threshold=80,
    use_fullframe=False,
    brightness_threshold=None,
    similar_image_threshold=100,
    similar_pixel_threshold=25,
    frames_to_skip=1,
    crop_x=None,
    crop_y=None,
    crop_width=None,
    crop_height=None,
) -> None:
    with open(file_path, "w+", encoding="utf-8") as f:
        f.write(
            get_subtitles(
                video_path,
                time_start,
                time_end,
                conf_threshold,
                sim_threshold,
                use_fullframe,
                brightness_threshold,
                similar_image_threshold,
                similar_pixel_threshold,
                frames_to_skip,
                crop_x,
                crop_y,
                crop_width,
                crop_height,
            )
        )
