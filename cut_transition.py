import glob
import logging
import multiprocessing as mp
import os
import time
from itertools import cycle

import numpy as np
import pandas as pd
from transnetv2 import TransNetV2

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

model = None


def init_worker(gpus):
    global model
    if not gpus.empty():
        gpu_id = gpus.get()
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        logger.info(f"Using GPU {gpu_id} on pid {os.getpid()}")
    else:
        logger.info(f"Using CPU only on pid {os.getpid()}")

    model = TransNetV2()


def inference_video(video_lst):
    for video in video_lst:
        video_id = os.path.splitext(os.path.basename(video))[0]
        csv_file_name = f"./log_csv/AIC_transition_stats_{video_id}.csv"
        if not os.path.exists(csv_file_name):
            video_bounding_transtion_dict = {
                "video_id": [],
                "transition_id": [],
                "frame_start": [],
                "frame_end": [],
            }
            _, single_frame_predictions, _ = model.predict_video(video)
            transition_frame_lst = model.predictions_to_scenes(single_frame_predictions)

            for idx, transition_frame in enumerate(transition_frame_lst):
                video_bounding_transtion_dict["video_id"].append(video_id)
                video_bounding_transtion_dict["transition_id"].append(
                    video_id + "_T" + str(idx).zfill(6)
                )
                video_bounding_transtion_dict["frame_start"].append(transition_frame[0])
                video_bounding_transtion_dict["frame_end"].append(transition_frame[-1])

            video_bounding_transtion_df = pd.DataFrame(video_bounding_transtion_dict)
            video_bounding_transtion_df.to_csv(csv_file_name, index=False)


def run_inference_in_process_pool(input_paths, num_process):
    # If GPUs are available, create a queue that loops through the GPU IDs.
    # For example, if there are 4 worker processes and 4 GPUs, the queue contains [0, 1, 2, 3]
    # If there are 4 worker processes and 2 GPUs the queue contains [0, 1, 0 ,1]
    gpus = [2,3,4, 7]
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
    process_pool.map(inference_video, input_paths)

    logger.info(
        "Processed {} images on {} processes for {:10.4f} seconds ".format(
            len(input_paths), num_process, time.time() - start
        )
    )


if __name__ == "__main__":
    video_path_batch_split = np.array_split(glob.glob("/app/data-batch-2/videos/*.mp4"), 4)

    run_inference_in_process_pool(video_path_batch_split, 4)
