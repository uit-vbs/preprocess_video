import glob
import logging
import multiprocessing as mp
import os
import statistics
import time
from itertools import cycle
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from lavis.models import load_model_and_preprocess
from PIL import Image

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

model = None
vis_processors = None

DATA_DIR = Path("/app/data-batch-2/videos/")
SECOND_TO_SKIP = 3

def get_frame_by_index(cap: cv2.VideoCapture, frame_idx: int):
    cap.set(1, frame_idx)
    _, frame = cap.read()
    return frame

def init_worker(gpus):
    global model, vis_processors

    if not gpus.empty():
        gpu_id = gpus.get()
        logger.info(f"Using GPU {gpu_id} on pid {os.getpid()}")
        torch.cuda.set_device(gpu_id)
        device = torch.device("cuda")
    else:
        logger.info(f"Using CPU only on pid {os.getpid()}")
        device = torch.device("cpu")

    model, vis_processors, _ = load_model_and_preprocess(
        name="blip_feature_extractor",
        model_type="base",
        is_eval=True,
        device=device,
    )


def extract_feature_from_video(csv_file_list):
    for csv_file in csv_file_list:
        df = pd.read_csv(csv_file)

        if not os.path.exists(f'features/{df["video_id"].unique()[0]}.npz'):
            video = cv2.VideoCapture(
                str(DATA_DIR.joinpath(df["video_id"].unique()[0] + ".mp4"))
            )
            fps = int(video.get(cv2.CAP_PROP_FPS))

            transition_stats_dict = {
                "feature_frame_index": [],
                "original_frame_index": [],
            }

            logger.info(f'Extract features from {df["video_id"].unique()[0] + ".mp4"}')

            feature_lst = []
            for _, transition in df.iterrows():
                if (
                    transition["frame_end"] - transition["frame_start"]
                    <= fps * SECOND_TO_SKIP
                ):
                    median_index = int(
                        statistics.median(
                            (transition["frame_start"], transition["frame_end"])
                        )
                    )

                    transition_stats_dict["feature_frame_index"].append(median_index)
                    transition_stats_dict["original_frame_index"].append(
                        transition["transition_id"].split("_")[-1]
                    )

                    median_frame = Image.fromarray(get_frame_by_index(video, median_index))
                    image = vis_processors["eval"](median_frame).unsqueeze(0).cuda()

                    sample = {"image": image, "text_input": [""]}

                    feature_image = model.extract_features(sample, mode="image")

                    feature_lst.append(feature_image.image_embeds_proj.cpu().numpy())
                else:
                    for step in range(
                        int(transition["frame_start"]),
                        int(transition["frame_end"]),
                        fps * SECOND_TO_SKIP,
                    ):
                        transition_stats_dict["feature_frame_index"].append(step)
                        transition_stats_dict["original_frame_index"].append(
                            transition["transition_id"].split("_")[-1]
                        )

                        step_frame = Image.fromarray(get_frame_by_index(video, step))
                        image = vis_processors["eval"](step_frame).unsqueeze(0).cuda()

                        sample = {"image": image, "text_input": [""]}

                        feature_image = model.extract_features(sample, mode="image")

                        feature_lst.append(feature_image.image_embeds_proj.cpu().numpy())

            np.savez_compressed(
                f'features/{df["video_id"].unique()[0]}.npz',
                feature_lst=feature_lst,
            )

            transition_frame_df = pd.DataFrame(transition_stats_dict)
            transition_frame_df.to_csv(
                f'features/{df["video_id"].unique()[0]}.csv', index=False
            )
        else:
            logger.info(f'{df["video_id"].unique()[0]}.mp4 is done')


def run_inference_in_process_pool(input_paths, num_process):
    # If GPUs are available, create a queue that loops through the GPU IDs.
    # For example, if there are 4 worker processes and 4 GPUs, the queue contains [0, 1, 2, 3]
    # If there are 4 worker processes and 2 GPUs the queue contains [0, 1, 0 ,1]
    gpus = [0, 2, 3, 4, 7]
    gpu_ids = mp.Queue()
    if len(gpus) > 0:
        gpu_id_cycle_iterator = cycle(gpus)
        for i in range(num_process):
            gpu_ids.put(next(gpu_id_cycle_iterator))

    # Initialize process pool
    process_pool = mp.Pool(
        processes=num_process, initializer=init_worker, initargs=(gpu_ids,), maxtasksperchild=100
    )

    start = time.time()
    # Feed inputs to process pool to do inference
    process_pool.map(extract_feature_from_video, input_paths)

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
        glob.glob("/app/TransNetV2/inference-pytorch/log_csv/*.csv"), 5
    )

    run_inference_in_process_pool(csv_path_batch_split, 5)
