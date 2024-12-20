# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import glob
import multiprocessing as mp
import os
import time
import cv2
import tqdm
import skimage.io as io

# from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger

from predictor import VisualizationDemo
from adet.config import get_cfg
import torch

from scipy.special import softmax
import numpy as np

# constants
WINDOW_NAME = "COCO detections"


def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.FCOS.INFERENCE_TH_TEST = args.confidence_threshold
    cfg.MODEL.MEInst.INFERENCE_TH_TEST = args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 Demo")
    parser.add_argument(
        "--config-file",
        default="../configs/SSIS/MS_R_101_BiFPN_with_offset_class_demo.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--webcam", action="store_true", help="Take inputs from webcam.")
    parser.add_argument("--video-input", help="Path to video file.")
    parser.add_argument("--input", default="./", help="A list of space separated input images")
    parser.add_argument(
        "--output",
        default="./res/",
        help="A file or directory to save output visualizations. "
             "If not given, will show output in an OpenCV window.",
    )

    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.1,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    logger = setup_logger()
    logger.info("Arguments: " + str(args))
    if os.path.exists(args.output) == False:
        os.mkdir(args.output)
    cfg = setup_cfg(args)
    args.input = [os.path.join(args.input, path) for path in os.listdir(args.input)]
    demo = VisualizationDemo(cfg)

    # Harry
    result = list()
    import pickle

    print("1111111", args.input)
    # Harry End

    if args.input:
        if os.path.isdir(args.input[0]):
            args.input = [os.path.join(args.input[0], fname) for fname in os.listdir(args.input[0])]
        elif len(args.input) == 1:
            args.input = glob.glob(os.path.expanduser(args.input[0]))
            assert args.input, "The input path(s) was not found"
        for path in tqdm.tqdm(args.input, disable=not args.output):
            # use PIL, to be consistent with evaluation
            torch.cuda.empty_cache()
            img = cv2.imread(path)
            start_time = time.time()
            with torch.no_grad():
                # instances, visualized_output = demo.run_on_image(img)
                # Harry
                try:
                    instances, visualized_output = demo.run_on_image(img)
                    height, weight, _ = img.shape
                    print("Harry 1: ", path, args.output)
                    print("Harry 2 num_instances: ", len(instances.scores))
                    print("Harry 3 image_size: ", (height, weight))
                    print("Harry 4 image_height: ", height)
                    print("Harry 5 image_width: ", weight)
                    print("Harry 6 pred_masks: ", instances.pred_masks)
                    print("Harry 7 pred_classes: ", instances.pred_classes)
                    print("Harry 8 pred_boxes: ", instances.pred_boxes.tensor)
                    print("Harry 9 scores: ", instances.scores)
                    print("Harry 10 fpn_levels: ", instances.fpn_levels)
                    print("Harry 11 offset: ", instances.offset)
                    print("Harry 12 normal: ", 1)

                    # print(instances.pred_classes)
                    # pred_classes_np = np.asarray(instances.pred_classes)
                    # track_embedding_obj = instances.track_embedding[pred_classes_np == 0]
                    # track_embedding_sha = instances.track_embedding[pred_classes_np == 1]
                    # print(track_embedding_obj.shape, track_embedding_sha.shape)
                    # test_track_embedding = np.concatenate(
                    #     (track_embedding_obj, track_embedding_sha), 1)
                    # print(test_track_embedding.shape)
                    # print(softmax(np.matmul(test_track_embedding, np.transpose(test_track_embedding, (1, 0)))))
                    # print(np.matmul(test_track_embedding, np.transpose(test_track_embedding, (1, 0))))

                    temp = dict()
                    name = path.split("/")[-2]
                    index = path.split("/")[-1]
                    temp['name'] = name
                    temp['index'] = index
                    temp['num_instances'] = len(instances.scores)
                    temp['image_size'] = (height, weight)
                    temp['image_height'] = height
                    temp['image_width'] = weight
                    temp['pred_masks'] = instances.pred_masks
                    temp['pred_classes'] = instances.pred_classes
                    temp['pred_boxes'] = instances.pred_boxes.tensor
                    temp['scores'] = instances.scores
                    temp['fpn_levels'] = instances.fpn_levels
                    temp['offset'] = instances.offset
                    temp['normal'] = 1
                    temp['track_embedding'] = instances.track_embedding
                    result.append(temp)
                except TypeError:
                    print("TypeError: cannot unpack non-iterable NoneType object")
                    height, weight, _ = img.shape
                    print("Harry 1: ", path)
                    print("Harry 2 num_instances: ", 0)
                    print("Harry 3 image_size: ", (height, weight))
                    print("Harry 4 image_height: ", height)
                    print("Harry 5 image_width: ", weight)
                    print("Harry 6 pred_masks: ", [])
                    print("Harry 7 pred_classes: ", [])
                    print("Harry 8 pred_boxes: ", [])
                    print("Harry 9 scores: ", [])
                    print("Harry 10 fpn_levels: ", [])
                    print("Harry 11 offset: ", [])
                    print("Harry 12 normal: ", 0)
                    temp = dict()
                    name = path.split("/")[-2]
                    index = path.split("/")[-1]
                    temp['name'] = name
                    temp['index'] = index
                    temp['num_instances'] = 0
                    temp['image_size'] = (height, weight)
                    temp['image_height'] = height
                    temp['image_width'] = weight
                    temp['pred_masks'] = []
                    temp['pred_classes'] = []
                    temp['pred_boxes'] = []
                    temp['scores'] = []
                    temp['fpn_levels'] = []
                    temp['offset'] = []
                    temp['normal'] = 0
                    temp['track_embedding'] = []
                    result.append(temp)

                    instances = []

                # Harry End
            logger.info(
                "{}: detected {} instances in {:.2f}s".format(
                    path, len(instances), time.time() - start_time
                )
            )

            # Harry
            if temp['normal'] == 0:
                os.system("cp -r " + path + " " + args.output + "/")

            elif temp['normal'] == 1:
                if args.output:

                    if os.path.isdir(args.output):
                        assert os.path.isdir(args.output), args.output
                        out_filename = os.path.join(args.output, os.path.basename(path))
                    else:
                        assert len(args.input) == 1, "Please specify a directory with args.output"
                        out_filename = args.output
                    visualized_output.save(out_filename)
                else:
                    cv2.imshow(WINDOW_NAME, visualized_output.get_image()[:, :, ::-1])
                    if cv2.waitKey(0) == 27:
                        break  # esc to quit
    elif args.webcam:
        assert args.input is None, "Cannot have both --input and --webcam!"
        cam = cv2.VideoCapture(0)
        for vis in tqdm.tqdm(demo.run_on_video(cam)):
            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
            cv2.imshow(WINDOW_NAME, vis)
            if cv2.waitKey(1) == 27:
                break  # esc to quit
        cv2.destroyAllWindows()
    elif args.video_input:
        video = cv2.VideoCapture(args.video_input)
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frames_per_second = video.get(cv2.CAP_PROP_FPS)
        num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        basename = os.path.basename(args.video_input)

        if args.output:
            if os.path.isdir(args.output):
                output_fname = os.path.join(args.output, basename)
                output_fname = os.path.splitext(output_fname)[0] + ".mkv"
            else:
                output_fname = args.output
            assert not os.path.isfile(output_fname), output_fname
            output_file = cv2.VideoWriter(
                filename=output_fname,
                # some installation of opencv may not support x264 (due to its license),
                # you can try other format (e.g. MPEG)
                fourcc=cv2.VideoWriter_fourcc(*"x264"),
                fps=float(frames_per_second),
                frameSize=(width, height),
                isColor=True,
            )
        assert os.path.isfile(args.video_input)
        for vis_frame in tqdm.tqdm(demo.run_on_video(video), total=num_frames):
            if args.output:
                output_file.write(vis_frame)
            else:
                cv2.namedWindow(basename, cv2.WINDOW_NORMAL)
                cv2.imshow(basename, vis_frame)
                if cv2.waitKey(1) == 27:
                    break  # esc to quit
        video.release()
        if args.output:
            output_file.release()
        else:
            cv2.destroyAllWindows()

    # Harry
    # to_path = "/content/mydrive/MyDrive/CUHK/projectX/Dataset_Selected_Clipped_Segmented/predicted_instance/"
    to_path = "/content/"

    # 读取
    # import pickle
    # result_file = open(to_path + 'instance_file.pickle','rb')
    # result_ = pickle.load(result_file)
    # result_file.close()

    # result_.extend(result)

    result_file = open(to_path + name + '.pickle', 'wb')
    pickle.dump(result, result_file, protocol=pickle.HIGHEST_PROTOCOL)
    result_file.close()
    # Harry End
