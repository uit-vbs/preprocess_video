import glob
import json
import logging
import torch.multiprocessing as mp
import os
import time
import random
from itertools import cycle
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from pipeline import Pipeline, init_args
from tool.config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

pipeline = None

config = Config("/app/vietnamese-ocr-toolbox/tool/config/configs.yaml")
args = init_args()

DATA_DIR = Path("/app/aic/videos")


def get_frame_by_index(cap: cv2.VideoCapture, frame_idx: int):
    cap.set(1, frame_idx)
    _, frame = cap.read()
    return frame


def init_worker(gpus):
    global pipeline

    if not gpus.empty():
        gpu_id = gpus.get()
        logger.info(f"Using GPU {gpu_id} on pid {os.getpid()}")
        torch.cuda.set_device(gpu_id)
        device = torch.device("cuda")
    else:
        logger.info(f"Using CPU only on pid {os.getpid()}")
        device = torch.device("cpu")

    pipeline = Pipeline(args, config, device, gpu_id)


def run_ocr_from_keyframe(csv_file_list):
    for csv_file in csv_file_list:
        video_id = os.path.splitext(os.path.basename(csv_file))[0]
        if not os.path.exists(
            f"/app/TransNetV2/inference-pytorch/features/OCR_result_{video_id}.json"
        ):
            df = pd.read_csv(csv_file)
            video = cv2.VideoCapture(str(DATA_DIR.joinpath(video_id + ".mp4")))

            ocr_info_dict = {
                "video_id": [],
                "transition_ocr_result": [],
            }

            ocr_info_dict["video_id"].append(video_id)
            grouped_df = df.groupby(by=["original_frame_index"])
            for key, _ in tqdm(grouped_df, desc=f'Running OCR for {video_id + ".mp4"}', leave=False):
                df_group_by_frame = grouped_df.get_group(key)
                transition_info_dict = {"transition_id": key, "ocr_result": []}
                for feature_idx in df_group_by_frame["feature_frame_index"]:
                    keyframe = get_frame_by_index(video, feature_idx)
                    # cv2.imwrite(f'{video_id}/{video_id}_{key}_{feature_idx}.png', keyframe)
                    results = pipeline.start(keyframe)

                    text_coordinates = [res[0] for res in results]
                    texts = [res[1][0] for res in results]

                    for coor, text in zip(text_coordinates, texts):
                        transition_info_dict["ocr_result"].append(
                            {
                                "coordinate": coor,
                                "text": text,
                                "feature_frame_index": feature_idx,
                            }
                        )

                ocr_info_dict["transition_ocr_result"].append(transition_info_dict)

            with open(
                f"/app/TransNetV2/inference-pytorch/features/OCR_result_{video_id}.json",
                "w",
                encoding="utf-8",
            ) as f:
                json.dump(ocr_info_dict, f)
        else:
            logger.info(f"{video_id}.mp4 is done.")

        time.sleep(random.randrange(0, 5))


def run_inference_in_process_pool(input_paths, num_process):
    # If GPUs are available, create a queue that loops through the GPU IDs.
    # For example, if there are 4 worker processes and 4 GPUs, the queue contains [0, 1, 2, 3]
    # If there are 4 worker processes and 2 GPUs the queue contains [0, 1, 0 ,1]
    gpus = [0, 3, 5, 6, 7]
    gpu_ids = mp.Queue()
    if len(gpus) > 0:
        gpu_id_cycle_iterator = cycle(gpus)
        for i in range(num_process):
            gpu_ids.put(next(gpu_id_cycle_iterator))

    # Initialize process pool
    process_pool = mp.Pool(
        processes=num_process,
        initializer=init_worker,
        initargs=(gpu_ids,),
        maxtasksperchild=80,
    )

    start = time.time()
    # Feed inputs to process pool to do inference
    process_pool.map(run_ocr_from_keyframe, input_paths)

    logger.info(
        "Processed {} batches on {} processes for {:10.4f} seconds ".format(
            len(input_paths), num_process, time.time() - start
        )
    )


if __name__ == "__main__":
    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        pass
    csv_path_batch_split = np.array_split(
        glob.glob("/app/TransNetV2/inference-pytorch/features/*.csv"), 5
    )

    run_inference_in_process_pool(csv_path_batch_split, 5)
