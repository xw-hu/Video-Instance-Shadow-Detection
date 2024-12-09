import argparse
import glob
import multiprocessing as mp
import os
import time
import cv2
import tqdm
from numpy.linalg import norm
from detectron2.utils.logger import setup_logger

from visd.demo.predictor import VisualizationDemo
from adet.config import get_cfg
import torch
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.special import softmax
from tools.painter import mask_painter
import re
from PIL import Image

def natural_sort_key(string):
    """
    Generate a key for natural sorting.
    Splits the string into parts of digits and non-digits for proper ordering.
    """
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', string)]

def generate_video_from_frames(image_folder, output_video_path, frame_rate=30):
    """
    Generate a video from image frames stored in a folder.

    Args:
        image_folder (str): Path to the folder containing image frames.
        output_video_path (str): Path to save the generated video.
        frame_rate (int): Frame rate for the output video. Default is 30 fps.
    """
    # Get all image paths in the folder
    image_paths = [os.path.join(image_folder, img) for img in os.listdir(image_folder) if img.endswith(('.png', '.jpg', '.jpeg'))]
    
    # Sort image paths naturally
    image_paths = sorted(image_paths, key=natural_sort_key)

    # Ensure there are images to process
    if not image_paths:
        raise ValueError("No images found in the specified folder.")

    # Read the first image to determine video properties
    first_image = cv2.imread(image_paths[0])
    if first_image is None:
        raise ValueError("Unable to read the first image.")

    height, width, channels = first_image.shape
    size = (width, height)

    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, frame_rate, size)

    # Write each image to the video
    for image_path in image_paths:
        frame = cv2.imread(image_path)
        if frame is not None:
            out.write(frame)
        else:
            print(f"Warning: Unable to read image {image_path}, skipping...")

    # Release the video writer
    out.release()

    print(f"Video successfully saved to {output_video_path}")

def dilate_mask(mask, kernel_size=3, iterations=3):
    """
    Dilate the mask.
    Args:
        mask (numpy array): The mask to be dilated.
        kernel_size (int): Size of the kernel. Defaults to 3.
        iterations (int): Number of times dilation is applied. Defaults to 1.
    
    Returns:
        numpy array: Dilated mask.
    """
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    dilated_mask = cv2.dilate(mask, kernel, iterations=iterations)
    return dilated_mask

def extract_number(filename):
    # This regular expression extracts the number from the filename
    match = re.search(r'(\d+)', filename)
    if match:
        return int(match.group(1))
    return 0  # Default value if no number is found.

def setup_cfg(config_file):
    # load config from file and command-line arguments
    cfg = get_cfg()
    cfg.merge_from_file(config_file)
    # cfg.merge_from_list(args.opts)
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = 0.1 #args.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.1 #args.confidence_threshold
    cfg.MODEL.FCOS.INFERENCE_TH_TEST = 0.1 #args.confidence_threshold
    cfg.MODEL.MEInst.INFERENCE_TH_TEST = 0.1 #args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = 0.1 #args.confidence_threshold
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 Demo")
    parser.add_argument("--input-name", help="Path to input image, directory, or video file.")
    parser.add_argument("--output-name", default="test", help="Directory to save the output.")
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.1,
        help="Minimum score for instance predictions to be shown",
    )
    return parser



class VisdTracker():
    def __init__(self, sigma_low = 0.0, sigma_high = 0.5, sigma_score = 0.15, t_min = 2, 
                    sigma_score_scale = 1.5, momentum = 0, cos_sim = 1, iou_decay_rate = .95, gap_decay_rate = .95, 
                    iou_nms_th = 0.1, track_mode = 2, find_missing = True, backward_track = True):
        self.sigma_low = sigma_low
        self.sigma_high = sigma_high
        self.sigma_score = sigma_score
        self.t_min = t_min
        self.sigma_score_scale = sigma_score_scale
        self.momentum = momentum
        self.cos_sim = cos_sim
        self.iou_decay_rate = iou_decay_rate
        self.gap_decay_rate = gap_decay_rate
        self.iou_nms_th = iou_nms_th
        self.track_mode = track_mode
        self.find_missing = find_missing
        self.backward_track = backward_track

        self.config_file = "./visd/configs/SSIS/CondInst_R_101_BiFPN_3x_sem_with_offset_class_Demo.yaml"

        self.COCO_CATEGORIES = [
            {"color": [220, 20, 60], "isthing": 1, "id": 1, "name": "person"},
            {"color": [119, 11, 32], "isthing": 1, "id": 2, "name": "bicycle"},
            {"color": [0, 0, 142], "isthing": 1, "id": 3, "name": "car"},
            {"color": [0, 0, 230], "isthing": 1, "id": 4, "name": "motorcycle"},
            {"color": [106, 0, 228], "isthing": 1, "id": 5, "name": "airplane"},
            {"color": [0, 60, 100], "isthing": 1, "id": 6, "name": "bus"},
            {"color": [0, 80, 100], "isthing": 1, "id": 7, "name": "train"},
            {"color": [0, 0, 70], "isthing": 1, "id": 8, "name": "truck"},
            {"color": [0, 0, 192], "isthing": 1, "id": 9, "name": "boat"},
            {"color": [250, 170, 30], "isthing": 1, "id": 10, "name": "traffic light"},
            {"color": [100, 170, 30], "isthing": 1, "id": 11, "name": "fire hydrant"},
            {"color": [220, 220, 0], "isthing": 1, "id": 13, "name": "stop sign"},
            {"color": [175, 116, 175], "isthing": 1, "id": 14, "name": "parking meter"},
            {"color": [250, 0, 30], "isthing": 1, "id": 15, "name": "bench"},
            {"color": [165, 42, 42], "isthing": 1, "id": 16, "name": "bird"},
            {"color": [255, 77, 255], "isthing": 1, "id": 17, "name": "cat"},
            {"color": [0, 226, 252], "isthing": 1, "id": 18, "name": "dog"},
            {"color": [182, 182, 255], "isthing": 1, "id": 19, "name": "horse"},
            {"color": [0, 82, 0], "isthing": 1, "id": 20, "name": "sheep"},
            {"color": [120, 166, 157], "isthing": 1, "id": 21, "name": "cow"},
            {"color": [110, 76, 0], "isthing": 1, "id": 22, "name": "elephant"},
            {"color": [174, 57, 255], "isthing": 1, "id": 23, "name": "bear"},
            {"color": [199, 100, 0], "isthing": 1, "id": 24, "name": "zebra"},
            {"color": [72, 0, 118], "isthing": 1, "id": 25, "name": "giraffe"},
            {"color": [255, 179, 240], "isthing": 1, "id": 27, "name": "backpack"},
            {"color": [0, 125, 92], "isthing": 1, "id": 28, "name": "umbrella"},
            {"color": [209, 0, 151], "isthing": 1, "id": 31, "name": "handbag"},
            {"color": [188, 208, 182], "isthing": 1, "id": 32, "name": "tie"},
            {"color": [0, 220, 176], "isthing": 1, "id": 33, "name": "suitcase"},
            {"color": [255, 99, 164], "isthing": 1, "id": 34, "name": "frisbee"},
            {"color": [92, 0, 73], "isthing": 1, "id": 35, "name": "skis"},
            {"color": [133, 129, 255], "isthing": 1, "id": 36, "name": "snowboard"},
            {"color": [78, 180, 255], "isthing": 1, "id": 37, "name": "sports ball"},
            {"color": [0, 228, 0], "isthing": 1, "id": 38, "name": "kite"},
            {"color": [174, 255, 243], "isthing": 1, "id": 39, "name": "baseball bat"},
            {"color": [45, 89, 255], "isthing": 1, "id": 40, "name": "baseball glove"},
            {"color": [134, 134, 103], "isthing": 1, "id": 41, "name": "skateboard"},
            {"color": [145, 148, 174], "isthing": 1, "id": 42, "name": "surfboard"},
            {"color": [255, 208, 186], "isthing": 1, "id": 43, "name": "tennis racket"},
            {"color": [197, 226, 255], "isthing": 1, "id": 44, "name": "bottle"},
            {"color": [171, 134, 1], "isthing": 1, "id": 46, "name": "wine glass"},
            {"color": [109, 63, 54], "isthing": 1, "id": 47, "name": "cup"},
            {"color": [207, 138, 255], "isthing": 1, "id": 48, "name": "fork"},
            {"color": [151, 0, 95], "isthing": 1, "id": 49, "name": "knife"},
            {"color": [9, 80, 61], "isthing": 1, "id": 50, "name": "spoon"},
            {"color": [84, 105, 51], "isthing": 1, "id": 51, "name": "bowl"},
            {"color": [74, 65, 105], "isthing": 1, "id": 52, "name": "banana"},
            {"color": [166, 196, 102], "isthing": 1, "id": 53, "name": "apple"},
            {"color": [208, 195, 210], "isthing": 1, "id": 54, "name": "sandwich"},
            {"color": [255, 109, 65], "isthing": 1, "id": 55, "name": "orange"},
            {"color": [0, 143, 149], "isthing": 1, "id": 56, "name": "broccoli"},
            {"color": [179, 0, 194], "isthing": 1, "id": 57, "name": "carrot"},
            {"color": [209, 99, 106], "isthing": 1, "id": 58, "name": "hot dog"},
            {"color": [5, 121, 0], "isthing": 1, "id": 59, "name": "pizza"},
            {"color": [227, 255, 205], "isthing": 1, "id": 60, "name": "donut"},
            {"color": [147, 186, 208], "isthing": 1, "id": 61, "name": "cake"},
            {"color": [153, 69, 1], "isthing": 1, "id": 62, "name": "chair"},
            {"color": [3, 95, 161], "isthing": 1, "id": 63, "name": "couch"},
            {"color": [163, 255, 0], "isthing": 1, "id": 64, "name": "potted plant"},
            {"color": [119, 0, 170], "isthing": 1, "id": 65, "name": "bed"},
            {"color": [0, 182, 199], "isthing": 1, "id": 67, "name": "dining table"},
            {"color": [0, 165, 120], "isthing": 1, "id": 70, "name": "toilet"},
            {"color": [183, 130, 88], "isthing": 1, "id": 72, "name": "tv"},
            {"color": [95, 32, 0], "isthing": 1, "id": 73, "name": "laptop"},
            {"color": [130, 114, 135], "isthing": 1, "id": 74, "name": "mouse"},
            {"color": [110, 129, 133], "isthing": 1, "id": 75, "name": "remote"},
            {"color": [166, 74, 118], "isthing": 1, "id": 76, "name": "keyboard"},
            {"color": [219, 142, 185], "isthing": 1, "id": 77, "name": "cell phone"},
            {"color": [79, 210, 114], "isthing": 1, "id": 78, "name": "microwave"},
            {"color": [178, 90, 62], "isthing": 1, "id": 79, "name": "oven"},
            {"color": [65, 70, 15], "isthing": 1, "id": 80, "name": "toaster"},
            {"color": [127, 167, 115], "isthing": 1, "id": 81, "name": "sink"},
            {"color": [59, 105, 106], "isthing": 1, "id": 82, "name": "refrigerator"},
            {"color": [142, 108, 45], "isthing": 1, "id": 84, "name": "book"},
            {"color": [196, 172, 0], "isthing": 1, "id": 85, "name": "clock"},
            {"color": [95, 54, 80], "isthing": 1, "id": 86, "name": "vase"},
            {"color": [128, 76, 255], "isthing": 1, "id": 87, "name": "scissors"},
            {"color": [201, 57, 1], "isthing": 1, "id": 88, "name": "teddy bear"},
            {"color": [246, 0, 122], "isthing": 1, "id": 89, "name": "hair drier"},
            {"color": [191, 162, 208], "isthing": 1, "id": 90, "name": "toothbrush"},
            {"color": [255, 255, 128], "isthing": 0, "id": 92, "name": "banner"},
            {"color": [147, 211, 203], "isthing": 0, "id": 93, "name": "blanket"},
            {"color": [150, 100, 100], "isthing": 0, "id": 95, "name": "bridge"},
            {"color": [168, 171, 172], "isthing": 0, "id": 100, "name": "cardboard"},
            {"color": [146, 112, 198], "isthing": 0, "id": 107, "name": "counter"},
            {"color": [210, 170, 100], "isthing": 0, "id": 109, "name": "curtain"},
            {"color": [92, 136, 89], "isthing": 0, "id": 112, "name": "door-stuff"},
            {"color": [218, 88, 184], "isthing": 0, "id": 118, "name": "floor-wood"},
            {"color": [241, 129, 0], "isthing": 0, "id": 119, "name": "flower"},
            {"color": [217, 17, 255], "isthing": 0, "id": 122, "name": "fruit"},
            {"color": [124, 74, 181], "isthing": 0, "id": 125, "name": "gravel"},
            {"color": [70, 70, 70], "isthing": 0, "id": 128, "name": "house"},
            {"color": [255, 228, 255], "isthing": 0, "id": 130, "name": "light"},
            {"color": [154, 208, 0], "isthing": 0, "id": 133, "name": "mirror-stuff"},
            {"color": [193, 0, 92], "isthing": 0, "id": 138, "name": "net"},
            {"color": [76, 91, 113], "isthing": 0, "id": 141, "name": "pillow"},
            {"color": [255, 180, 195], "isthing": 0, "id": 144, "name": "platform"},
            {"color": [106, 154, 176], "isthing": 0, "id": 145, "name": "playingfield"},
            {"color": [230, 150, 140], "isthing": 0, "id": 147, "name": "railroad"},
            {"color": [60, 143, 255], "isthing": 0, "id": 148, "name": "river"},
            {"color": [128, 64, 128], "isthing": 0, "id": 149, "name": "road"},
            {"color": [92, 82, 55], "isthing": 0, "id": 151, "name": "roof"},
            {"color": [254, 212, 124], "isthing": 0, "id": 154, "name": "sand"},
            {"color": [73, 77, 174], "isthing": 0, "id": 155, "name": "sea"},
            {"color": [255, 160, 98], "isthing": 0, "id": 156, "name": "shelf"},
            {"color": [255, 255, 255], "isthing": 0, "id": 159, "name": "snow"},
            {"color": [104, 84, 109], "isthing": 0, "id": 161, "name": "stairs"},
            {"color": [169, 164, 131], "isthing": 0, "id": 166, "name": "tent"},
            {"color": [225, 199, 255], "isthing": 0, "id": 168, "name": "towel"},
            {"color": [137, 54, 74], "isthing": 0, "id": 171, "name": "wall-brick"},
            {"color": [135, 158, 223], "isthing": 0, "id": 175, "name": "wall-stone"},
            {"color": [7, 246, 231], "isthing": 0, "id": 176, "name": "wall-tile"},
            {"color": [107, 255, 200], "isthing": 0, "id": 177, "name": "wall-wood"},
            {"color": [58, 41, 149], "isthing": 0, "id": 178, "name": "water-other"},
            {"color": [183, 121, 142], "isthing": 0, "id": 180, "name": "window-blind"},
            {"color": [255, 73, 97], "isthing": 0, "id": 181, "name": "window-other"},
            {"color": [107, 142, 35], "isthing": 0, "id": 184, "name": "tree-merged"},
            {"color": [190, 153, 153], "isthing": 0, "id": 185, "name": "fence-merged"},
            {"color": [146, 139, 141], "isthing": 0, "id": 186, "name": "ceiling-merged"},
            {"color": [70, 130, 180], "isthing": 0, "id": 187, "name": "sky-other-merged"},
            {"color": [134, 199, 156], "isthing": 0, "id": 188, "name": "cabinet-merged"},
            {"color": [209, 226, 140], "isthing": 0, "id": 189, "name": "table-merged"},
            {"color": [96, 36, 108], "isthing": 0, "id": 190, "name": "floor-other-merged"},
            {"color": [96, 96, 96], "isthing": 0, "id": 191, "name": "pavement-merged"},
            {"color": [64, 170, 64], "isthing": 0, "id": 192, "name": "mountain-merged"},
            {"color": [152, 251, 152], "isthing": 0, "id": 193, "name": "grass-merged"},
            {"color": [208, 229, 228], "isthing": 0, "id": 194, "name": "dirt-merged"},
            {"color": [206, 186, 171], "isthing": 0, "id": 195, "name": "paper-merged"},
            {"color": [152, 161, 64], "isthing": 0, "id": 196, "name": "food-other-merged"},
            {"color": [116, 112, 0], "isthing": 0, "id": 197, "name": "building-other-merged"},
            {"color": [0, 114, 143], "isthing": 0, "id": 198, "name": "rock-merged"},
            {"color": [102, 102, 156], "isthing": 0, "id": 199, "name": "wall-other-merged"},
            {"color": [250, 141, 255], "isthing": 0, "id": 200, "name": "rug-merged"},
        ]

        self.thing_colors = [k["color"] for k in self.COCO_CATEGORIES]

    def extract_frames_from_video(self, video_path):
        frame_paths = []
        cap = cv2.VideoCapture(video_path)
        frame_idx = 0

        # Create a temporary directory for storing frames
        temp_dir = os.path.join("./tmp", "video_frames")
        os.makedirs(temp_dir, exist_ok=True)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_path = os.path.join(temp_dir, f"frame_{frame_idx:04d}.jpg")
            cv2.imwrite(frame_path, frame)
            frame_paths.append(frame_path)
            frame_idx += 1

        cap.release()
        return frame_paths


    def pred_result_generate(self, input_path, output_path):
        mp.set_start_method("spawn", force=True)
        logger = setup_logger()
        config_file = self.config_file
        cfg = setup_cfg(config_file)
        input_path = [os.path.join(input_path, path) for path in os.listdir(input_path)]
        demo = VisualizationDemo(cfg)

        result = list()

        if input_path:
            if os.path.isdir(input_path[0]):
                input_path = [os.path.join(input_path[0], fname) for fname in os.listdir(input_path[0])]
            elif len(input_path) == 1:
                input_path = glob.glob(os.path.expanduser(input_path[0]))
                assert input_path, "The input path(s) was not found"

            input_path = sorted(input_path, key=extract_number)
            for path in tqdm.tqdm(input_path, disable=not output_path):
                # use PIL, to be consistent with evaluation
                torch.cuda.empty_cache()
                # print(path)
                img = cv2.imread(path)
                start_time = time.time()
                with torch.no_grad():
                    # instances, visualized_output = demo.run_on_image(img)
                    # Harry
                    try:
                        instances, visualized_output = demo.run_on_image(img)
                        # instances = instances[0]['instances']
                        height, weight, _ = img.shape
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

                        temp['rest_pred_masks'] = instances.rest_pred_masks
                        temp['rest_pred_asso_masks'] = instances.rest_pred_asso_masks
                        temp['rest_pred_boxes'] = instances.rest_pred_boxes.tensor
                        temp['rest_scores'] = instances.rest_scores
                        temp['rest_pred_classes'] = instances.rest_pred_classes
                        temp['rest_track_embeddings'] = instances.rest_track_embeddings
                        result.append(temp)
                    except TypeError:
                        print("TypeError: cannot unpack non-iterable NoneType object")
                        height, weight, _ = img.shape
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

                        temp['rest_pred_masks'] = []
                        temp['rest_pred_asso_masks'] = []
                        temp['rest_pred_boxes'] = []
                        temp['rest_scores'] = []
                        temp['rest_pred_classes'] = []
                        temp['rest_track_embeddings'] = []
                        result.append(temp)
                    # Harry End

        return result


    # Actually box IoU
    def compute_ious(self, pred, target):
        """
        Args:
            pred: Nx4 predicted bounding boxes
            target: Nx4 target bounding boxes
            Both are in the form of FCOS prediction (l, t, r, b)
        """
        pred_left = pred[:, 0]
        pred_top = pred[:, 1]
        pred_right = pred[:, 2]
        pred_bottom = pred[:, 3]

        target_left = target[:, 0]
        target_top = target[:, 1]
        target_right = target[:, 2]
        target_bottom = target[:, 3]

        target_aera = (target_left + target_right) * \
                    (target_top + target_bottom)
        pred_aera = (pred_left + pred_right) * \
                    (pred_top + pred_bottom)

        w_intersect = min(pred_left, target_left) + \
                    min(pred_right, target_right)
        h_intersect = min(pred_bottom, target_bottom) + \
                    min(pred_top, target_top)

        g_w_intersect = max(pred_left, target_left) + \
                        max(pred_right, target_right)
        g_h_intersect = max(pred_bottom, target_bottom) + \
                        max(pred_top, target_top)
        ac_uion = g_w_intersect * g_h_intersect

        area_intersect = w_intersect * h_intersect
        area_union = target_aera + pred_aera - area_intersect

        ious = (area_intersect + 1.0) / (area_union + 1.0)
        # gious = ious - (ac_uion - area_union) / ac_uion

        return ious, None #, gious

    # Mask IoU
    def binaryMaskIOU(self, mask1, mask2):
        mask1_area = np.count_nonzero(mask1)
        mask2_area = np.count_nonzero(mask2)
        intersection = np.count_nonzero(np.logical_and(mask1, mask2))
        if mask1_area + mask2_area - intersection == 0:
            iou = 0
        else:
            iou = intersection / (mask1_area + mask2_area - intersection)
        return iou


    # Mask IoU
    # def videoBinaryMaskIOU(self, mask1, mask2):
    #     mask1_area = 0
    #     mask2_area = 0
    #     intersection = 0

    #     for mask_index in range(len(mask1)):
    #         mask1_area += np.count_nonzero(mask1[mask_index])
    #         mask2_area += np.count_nonzero(mask2[mask_index])
    #         intersection += np.count_nonzero(np.logical_and(mask1[mask_index], mask2[mask_index]))

    #     if mask1_area + mask2_area - intersection == 0:
    #         iou = 0
    #     else:
    #         iou = intersection / (mask1_area + mask2_area - intersection)
    #     return iou


    def SSIS_Tracker(self, D, match_mode, sigma_low, sigma_high, sigma_score, t_min, sigma_score_scale, momentum, cos_sim,
                    iou_decay_rate, gap_decay_rate, find_missing, backward_track):
        # match_mode: 0 is object mask, 1 is shaodw mask, 2 is asocciated mask

        Ta = list()  # active tracks
        Tf = list()  # finished tracks
        Ta_lastMatch = []  # The time of the last match for each trajectory
        Ta_MatchMode = dict()
        # Preprocessing: Select and process data based on track mode
        for frame_index in range(len(D)):
            used_masks_obj = D[frame_index]['obj_boxes']
            used_masks_sha = D[frame_index]['sha_boxes']
            if match_mode == 0:
                used_scores = D[frame_index]['obj_scores']
                used_track_embeddings = D[frame_index]['obj_track_embeddings']
            elif match_mode == 1:
                used_scores = D[frame_index]['sha_scores']
                used_track_embeddings = D[frame_index]['sha_track_embeddings']
            elif match_mode == 2:
                used_scores = D[frame_index]['asso_scores']
                used_track_embeddings = D[frame_index]['asso_track_embeddings']

            # Preprocessing: Filter out low confidence masks
            temp_scores = []
            temp_masks_obj = []
            temp_masks_sha = []
            temp_track_embeddings = []
            for instance_index in range(len(used_scores)):
                if used_scores[instance_index] >= sigma_low:
                    temp_scores.append(used_scores[instance_index])
                    temp_masks_obj.append(used_masks_obj[instance_index])
                    temp_masks_sha.append(used_masks_sha[instance_index])
                    temp_track_embeddings.append(used_track_embeddings[instance_index])
            used_scores = temp_scores
            used_masks_obj = temp_masks_obj
            used_masks_sha = temp_masks_sha
            used_track_embeddings = temp_track_embeddings

            D[frame_index]['used_scores'] = used_scores
            D[frame_index]['used_masks_obj'] = used_masks_obj
            D[frame_index]['used_masks_sha'] = used_masks_sha
            D[frame_index]['used_track_embeddings'] = used_track_embeddings

        # Start
        Rest_frame_instance_matched = []
        for frame_index in range(len(D)):
            print("Frame: ", frame_index + 1)
            used_instance_index = []
            new_Tf = []
            new_Ta = []
            Ta_MatchQueueIndexs = []  # The time of the last match for each trajectory
            Rest_frame_instance_matched.append([])

            #########################################################
            D[frame_index]['used_sim_score_sotfmax_instance'] = list()
            D[frame_index]['used_sim_score_instance'] = list()
            if len(Ta) != 0:
                for instance_index, instance_mask in enumerate(D[frame_index]["used_masks_obj"]):
                    sim_vector = []
                    for t_i_index in range(len(Ta)):
                        t_i_latest_track_embedding = Ta[t_i_index]["used_track_embedding"][-1]
                        temp_track_embedding = D[frame_index]['used_track_embeddings'][instance_index]
                        # '''
                        if Ta_MatchMode[t_i_index] == 0:
                            t_i_latest_track_embedding = Ta[t_i_index]["obj_track_embedding"][-1]
                            temp_track_embedding = D[frame_index]['obj_track_embeddings'][instance_index]
                        elif Ta_MatchMode[t_i_index] == 1:
                            t_i_latest_track_embedding = Ta[t_i_index]["sha_track_embedding"][-1]
                            temp_track_embedding = D[frame_index]['sha_track_embeddings'][instance_index]
                        # '''
                        if cos_sim == 0:
                            temp_sim = np.matmul(np.reshape(t_i_latest_track_embedding, (1, -1)),
                                                np.reshape(temp_track_embedding, (-1, 1)))
                        elif cos_sim == 1:
                            # Harry New
                            A = np.reshape(t_i_latest_track_embedding, (1, -1))
                            B = np.reshape(temp_track_embedding, (-1, 1))
                            temp_sim = np.dot(A, B) / (norm(A) * norm(B))
                        sim_vector.append(temp_sim.item())
                    sim_vector = np.array(sim_vector)
                    sim_vector_softmax = softmax(sim_vector)

                    D[frame_index]['used_sim_score_sotfmax_instance'].append(list(sim_vector_softmax))
                    D[frame_index]['used_sim_score_instance'].append(list(sim_vector))
            else:
                for instance_index, instance_mask in enumerate(D[frame_index]["used_masks_obj"]):
                    D[frame_index]['used_sim_score_sotfmax_instance'].append([])
                    D[frame_index]['used_sim_score_instance'].append(list([]))

            if len(D[frame_index]["used_masks_obj"]) != 0:
                for t_i_index in range(len(Ta)):
                    Ta[t_i_index]['used_sim_score_sotfmax_trajectory'] = list()
                    sim_vector = []
                    for instance_index, instance_mask in enumerate(D[frame_index]["used_masks_obj"]):

                        t_i_latest_track_embedding = Ta[t_i_index]["used_track_embedding"][-1]
                        temp_track_embedding = D[frame_index]['used_track_embeddings'][instance_index]
                        # '''
                        if Ta_MatchMode[t_i_index] == 0:
                            t_i_latest_track_embedding = Ta[t_i_index]["obj_track_embedding"][-1]
                            temp_track_embedding = D[frame_index]['obj_track_embeddings'][instance_index]
                        elif Ta_MatchMode[t_i_index] == 1:
                            t_i_latest_track_embedding = Ta[t_i_index]["sha_track_embedding"][-1]
                            temp_track_embedding = D[frame_index]['sha_track_embeddings'][instance_index]
                        # '''
                        if cos_sim == 0:
                            temp_sim = np.matmul(np.reshape(t_i_latest_track_embedding, (1, -1)),
                                                np.reshape(temp_track_embedding, (-1, 1)))
                        elif cos_sim == 1:
                            # Harry New
                            A = np.reshape(t_i_latest_track_embedding, (1, -1))
                            B = np.reshape(temp_track_embedding, (-1, 1))
                            temp_sim = np.dot(A, B) / (norm(A) * norm(B))
                        sim_vector.append(temp_sim.item())
                    sim_vector = np.array(sim_vector)
                    sim_vector_softmax = softmax(sim_vector)

                    Ta[t_i_index]['used_sim_score_sotfmax_trajectory'].extend(list(sim_vector_softmax))
            else:
                for t_i_index in range(len(Ta)):
                    Ta[t_i_index]['used_sim_score_sotfmax_trajectory'] = list()
                    Ta[t_i_index]['used_sim_score_sotfmax_trajectory'].extend([])
            #########################################################
            cost_matrix = []
            score_matrix = []
            sim_matrix = []
            sim_dir_matrix = []
            for instance_index, instance_mask in enumerate(D[frame_index]["used_masks_obj"]):
                cost_matrix_row = []
                score_matrix_row = []
                sim_matrix_row = []
                sim_dir_matrix_row = []
                for t_i_index in range(len(Ta)):
                    t_i_latest_mask_obj = Ta[t_i_index]["used_masks_obj"][-1]
                    t_i_latest_mask_sha = Ta[t_i_index]["used_masks_sha"][-1]
                    last_match_gap = Ta_lastMatch[t_i_index]  # if Ta_lastMatch[t_i_index] < 3 else 3
                    temp_confidence_score = D[frame_index]['used_scores'][instance_index]
                    temp_iou_obj, _ = self.compute_ious(torch.tensor(t_i_latest_mask_obj).reshape(1, -1),
                                                torch.tensor(D[frame_index]["used_masks_obj"][instance_index]).reshape(1, -1))
                    temp_iou_obj *= pow(iou_decay_rate, last_match_gap)
                    temp_iou_sha, _ = self.compute_ious(torch.tensor(t_i_latest_mask_sha).reshape(1, -1),
                                                torch.tensor(D[frame_index]["used_masks_sha"][instance_index]).reshape(1, -1))
                    temp_iou_sha *= pow(iou_decay_rate, last_match_gap)
                    if match_mode in [2]:
                        temp_iou = (temp_iou_obj + temp_iou_sha) / 2
                    elif match_mode == 0:
                        temp_iou = temp_iou_obj
                    elif match_mode == 1:
                        temp_iou = temp_iou_sha

                    # '''
                    if Ta_MatchMode[t_i_index] == 0:
                        temp_iou = temp_iou_obj
                        temp_confidence_score = D[frame_index]['obj_scores'][instance_index]
                    elif Ta_MatchMode[t_i_index] == 1:
                        temp_iou = temp_iou_sha
                        temp_confidence_score = D[frame_index]['sha_scores'][instance_index]
                    # '''

                    # Harry New

                    temp_sim = (D[frame_index]['used_sim_score_sotfmax_instance'][instance_index][t_i_index] +
                                Ta[t_i_index]['used_sim_score_sotfmax_trajectory'][instance_index]) / 2
                    temp_sim = temp_sim * pow(gap_decay_rate, last_match_gap)
                    temp_score = temp_sim.item() + 0.5 * temp_iou.item() + 0.25 * temp_confidence_score
                    temp_sim_dir = temp_sim.item() + temp_iou.item()

                    score_matrix_row.append([temp_score, temp_sim, temp_confidence_score, temp_iou])
                    cost_matrix_row.append(temp_score)
                    sim_matrix_row.append(temp_sim)
                    sim_dir_matrix_row.append(temp_sim_dir)

                cost_matrix.append(cost_matrix_row)
                score_matrix.append(score_matrix_row)
                sim_matrix.append(sim_matrix_row)
                sim_dir_matrix.append(sim_dir_matrix_row)
            if len(D[frame_index]["used_masks_obj"]) == 0:
                cost_matrix.append([])
                sim_matrix.append([])
                sim_dir_matrix.append([])

            row_ind, col_ind = linear_sum_assignment(sim_dir_matrix, maximize=True)

            if len(D[frame_index]["used_masks_obj"]) > 0 and len(Ta) > 0:
                sigma_score_final = sigma_score + 1 / (len(Ta) + 1) * sigma_score_scale
                # print("sigma_score_final: ", sigma_score_final, "len of Ta: ", len(Ta))
            else:
                sigma_score_final = sigma_score
                # print("no sigma_score_final: ", sigma_score_final, "len of Ta: ", len(Ta))

            # 20221002 For Missing Instances
            #########################################################
            if find_missing == True:
                D[frame_index]['rest_used_sim_score_sotfmax_instance'] = list()
                D[frame_index]['rest_used_sim_score_instance'] = list()

                len_rest = len(D[frame_index]['rest_track_embeddings'])
                temp_rest_all_tracking_embedding = D[frame_index]['rest_track_embeddings'].detach().cpu().numpy().tolist()
                temp_rest_all_tracking_embedding.extend(D[frame_index]['used_track_embeddings'])
                temp_rest_all_mask_obj = D[frame_index]['rest_boxes'].tolist()
                temp_rest_all_mask_obj.extend(D[frame_index]['used_masks_obj'])
                temp_rest_all_mask_sha = D[frame_index]['rest_boxes'].tolist()
                temp_rest_all_mask_sha.extend(D[frame_index]['used_masks_sha'])
                temp_rest_all_score = D[frame_index]['rest_scores'].detach().cpu().numpy().tolist()
                temp_rest_all_score.extend(D[frame_index]['used_scores'])
                if len(Ta) != 0:
                    for instance_index, instance_mask in enumerate(temp_rest_all_tracking_embedding):
                        sim_vector = []
                        for t_i_index in range(len(Ta)):
                            if instance_index < len_rest:
                                if D[frame_index]["rest_classes"][instance_index] == 0:
                                    t_i_latest_track_embedding = Ta[t_i_index]["obj_track_embedding"][-1]
                                elif D[frame_index]["rest_classes"][instance_index] == 1:
                                    t_i_latest_track_embedding = Ta[t_i_index]["sha_track_embedding"][-1]
                            else:
                                t_i_latest_track_embedding = Ta[t_i_index]["used_track_embedding"][-1]
                            temp_track_embedding = temp_rest_all_tracking_embedding[instance_index]
                            if cos_sim == 0:
                                temp_sim = np.matmul(np.reshape(t_i_latest_track_embedding, (1, -1)),
                                                    np.reshape(temp_track_embedding, (-1, 1)))
                            elif cos_sim == 1:
                                # Harry New
                                A = np.reshape(t_i_latest_track_embedding, (1, -1))
                                B = np.reshape(temp_track_embedding, (-1, 1))
                                temp_sim = np.dot(A, B) / (norm(A) * norm(B))
                            sim_vector.append(temp_sim.item())
                        sim_vector = np.array(sim_vector)
                        sim_vector_softmax = softmax(sim_vector)

                        D[frame_index]['rest_used_sim_score_sotfmax_instance'].append(list(sim_vector_softmax))
                        D[frame_index]['rest_used_sim_score_instance'].append(list(sim_vector))
                else:
                    for instance_index, instance_mask in enumerate(temp_rest_all_tracking_embedding):
                        D[frame_index]['rest_used_sim_score_sotfmax_instance'].append([])
                        D[frame_index]['rest_used_sim_score_instance'].append([])

                if len(temp_rest_all_tracking_embedding) != 0:
                    for t_i_index in range(len(Ta)):
                        Ta[t_i_index]['rest_used_sim_score_sotfmax_trajectory'] = list()
                        sim_vector = []
                        for instance_index, instance_mask in enumerate(temp_rest_all_tracking_embedding):
                            if instance_index < len_rest:
                                if D[frame_index]["rest_classes"][instance_index] == 0:
                                    t_i_latest_track_embedding = Ta[t_i_index]["obj_track_embedding"][-1]
                                elif D[frame_index]["rest_classes"][instance_index] == 1:
                                    t_i_latest_track_embedding = Ta[t_i_index]["sha_track_embedding"][-1]
                            else:
                                t_i_latest_track_embedding = Ta[t_i_index]["used_track_embedding"][-1]
                            temp_track_embedding = temp_rest_all_tracking_embedding[instance_index]

                            if cos_sim == 0:
                                temp_sim = np.matmul(np.reshape(t_i_latest_track_embedding, (1, -1)),
                                                    np.reshape(temp_track_embedding, (-1, 1)))
                            elif cos_sim == 1:
                                # Harry New
                                A = np.reshape(t_i_latest_track_embedding, (1, -1))
                                B = np.reshape(temp_track_embedding, (-1, 1))
                                temp_sim = np.dot(A, B) / (norm(A) * norm(B))
                            sim_vector.append(temp_sim.item())
                        sim_vector = np.array(sim_vector)
                        sim_vector_softmax = softmax(sim_vector)

                        Ta[t_i_index]['rest_used_sim_score_sotfmax_trajectory'].extend(list(sim_vector_softmax))
                else:
                    for t_i_index in range(len(Ta)):
                        Ta[t_i_index]['rest_used_sim_score_sotfmax_trajectory'] = list()
                        Ta[t_i_index]['rest_used_sim_score_sotfmax_trajectory'].extend([])

                rest_cost_matrix = []
                rest_score_matrix = []
                rest_sim_matrix = []
                rest_sim_dir_matrix = []
                for instance_index, instance_mask in enumerate(temp_rest_all_mask_obj):
                    cost_matrix_row = []
                    score_matrix_row = []
                    sim_matrix_row = []
                    sim_dir_matrix_row = []
                    for t_i_index in range(len(Ta)):
                        t_i_latest_mask_obj = Ta[t_i_index]["used_masks_obj"][-1]
                        t_i_latest_mask_sha = Ta[t_i_index]["used_masks_sha"][-1]
                        last_match_gap = Ta_lastMatch[t_i_index]  # if Ta_lastMatch[t_i_index] < 3 else 3
                        if instance_index < len_rest:
                            temp_iou_obj, _ = self.compute_ious(torch.tensor(t_i_latest_mask_obj).reshape(1, -1),
                                                        torch.tensor(temp_rest_all_mask_obj[instance_index]).reshape(1,-1))
                            temp_iou_obj *= pow(iou_decay_rate, last_match_gap)
                            temp_iou_sha, _ = self.compute_ious(torch.tensor(t_i_latest_mask_sha).reshape(1, -1),
                                                        torch.tensor(temp_rest_all_mask_sha[instance_index]).reshape(1,
                                                                                                                        -1))
                            temp_iou_sha *= pow(iou_decay_rate, last_match_gap)
                            if D[frame_index]["rest_classes"][instance_index] == 0:
                                temp_iou = temp_iou_obj
                            else:
                                temp_iou = temp_iou_sha
                        else:
                            temp_iou_obj, _ = self.compute_ious(torch.tensor(t_i_latest_mask_obj).reshape(1, -1),
                                                        torch.tensor(temp_rest_all_mask_obj[instance_index]).reshape(1, -1))
                            temp_iou_obj *= pow(iou_decay_rate, last_match_gap)
                            temp_iou_sha, _ = self.compute_ious(torch.tensor(t_i_latest_mask_sha).reshape(1, -1),
                                                        torch.tensor(temp_rest_all_mask_sha[instance_index]).reshape(1, -1))
                            temp_iou_sha *= pow(iou_decay_rate, last_match_gap)
                            if match_mode == 2:
                                temp_iou = (temp_iou_obj + temp_iou_sha) * 0.5
                            elif match_mode == 0:
                                temp_iou = temp_iou_obj
                            else:
                                temp_iou = temp_iou_sha

                        temp_confidence_score = temp_rest_all_score[instance_index]

                        temp_sim = (D[frame_index]['rest_used_sim_score_sotfmax_instance'][instance_index][t_i_index] +
                                    Ta[t_i_index]['rest_used_sim_score_sotfmax_trajectory'][instance_index]) / 2
                        temp_sim = temp_sim * pow(gap_decay_rate, last_match_gap)
                        temp_score = temp_sim.item() + 0.5 * temp_iou.item() + 0.25 * temp_confidence_score
                        temp_sim_dir = temp_sim.item() + temp_iou.item()
                        score_matrix_row.append([temp_score, temp_sim, temp_confidence_score, temp_iou])
                        if torch.is_tensor(temp_score):
                            temp_score = temp_score.item()
                        cost_matrix_row.append(temp_score)
                        sim_matrix_row.append(temp_sim)
                        sim_dir_matrix_row.append(temp_sim_dir)

                    rest_cost_matrix.append(cost_matrix_row)
                    rest_score_matrix.append(score_matrix_row)
                    rest_sim_matrix.append(sim_matrix_row)
                    rest_sim_dir_matrix.append(sim_dir_matrix_row)
                if len(temp_rest_all_tracking_embedding) == 0:
                    rest_cost_matrix.append([])
                    rest_sim_matrix.append([])
                    rest_sim_dir_matrix.append([])

                rest_row_ind, rest_col_ind = linear_sum_assignment(rest_sim_dir_matrix, maximize=True)
                if len(temp_rest_all_tracking_embedding) > 0 and len(Ta) > 0:
                    rest_sigma_score_final = sigma_score + 1 / (
                                len(Ta) + 1) * sigma_score_scale  # (1/len(Ta))*sigma_score_scale
                    # print("rest_sigma_score_final: ", rest_sigma_score_final, "len of Ta: ", len(Ta))
                else:
                    rest_sigma_score_final = sigma_score  # sigma_score_scale
                    # print("no rest_sigma_score_final: ", rest_sigma_score_final, "len of Ta: ", len(Ta))
            ########################################################
            for match_index in range(len(row_ind)):
                instance_match_index = row_ind[match_index]
                trajectory_match_index = col_ind[match_index]
                if cost_matrix[instance_match_index][trajectory_match_index] > sigma_score_final:

                    Ta_MatchMode[trajectory_match_index] = match_mode

                    # SSIS_v1
                    Ta[trajectory_match_index]['obj_track_embedding'].append(
                        D[frame_index]['obj_track_embeddings'][instance_match_index])  # e.g. embedding
                    Ta[trajectory_match_index]['sha_track_embedding'].append(
                        D[frame_index]['sha_track_embeddings'][instance_match_index])  # e.g. embedding
                    Ta[trajectory_match_index]['asso_track_embedding'].append(
                        D[frame_index]['asso_track_embeddings'][instance_match_index])  # e.g. embedding

                    Ta[trajectory_match_index]['obj_mask'].append(D[frame_index]['obj_masks'][instance_match_index])
                    Ta[trajectory_match_index]['sha_mask'].append(D[frame_index]['sha_masks'][instance_match_index])
                    Ta[trajectory_match_index]['asso_mask'].append(D[frame_index]['asso_masks'][instance_match_index])

                    Ta[trajectory_match_index]['obj_score'].append(D[frame_index]['obj_scores'][instance_match_index])
                    Ta[trajectory_match_index]['sha_score'].append(D[frame_index]['sha_scores'][instance_match_index])
                    Ta[trajectory_match_index]['asso_score'].append(D[frame_index]['asso_scores'][instance_match_index])

                    Ta[trajectory_match_index]['index'].append(frame_index)  # e.g. 0
                    Ta[trajectory_match_index]['frame_name'].append(D[frame_index]['index'])  # e.g. '00000.jpg'
                    Ta[trajectory_match_index]["used_score"].append(
                        D[frame_index]['used_scores'][instance_match_index])  # e.g. confidence score
                    Ta[trajectory_match_index]['used_masks_obj'].append(
                        D[frame_index]["used_masks_obj"][instance_match_index])  # e.g. mask
                    Ta[trajectory_match_index]['used_masks_sha'].append(
                        D[frame_index]["used_masks_sha"][instance_match_index])  # e.g. mask
                    new_use_emb = Ta[trajectory_match_index]['used_track_embedding'][-1] * momentum + \
                                D[frame_index]['used_track_embeddings'][instance_match_index] * (1 - momentum)
                    Ta[trajectory_match_index]['used_track_embedding'].append(new_use_emb)

                    used_instance_index.append(instance_match_index)
                    Ta_MatchQueueIndexs.append(trajectory_match_index)
                # else:
                    # print(instance_match_index, trajectory_match_index, "失败 ", "best_score: ",
                    #     score_matrix[instance_match_index][trajectory_match_index][0],
                    #     "best_sim: ", score_matrix[instance_match_index][trajectory_match_index][1],
                    #     "best_confidence_score: ", score_matrix[instance_match_index][trajectory_match_index][2],
                    #     "best_iou: ", score_matrix[instance_match_index][trajectory_match_index][3])

                    # print("-" * 100)


            # 20221002 For Missing Instances
            #########################################################
            # '''
            if find_missing == True:
                Rest_instance_matched = []
                for match_index in range(len(rest_row_ind)):
                    instance_match_index = rest_row_ind[match_index]
                    trajectory_match_index = rest_col_ind[match_index]
                    if trajectory_match_index not in Ta_MatchQueueIndexs and \
                            rest_cost_matrix[instance_match_index][trajectory_match_index] > rest_sigma_score_final and \
                            instance_match_index < len_rest:

                        Ta_MatchMode[trajectory_match_index] = D[frame_index]['rest_classes'][instance_match_index]

                        if D[frame_index]['rest_classes'][instance_match_index] == 0:
                            Ta[trajectory_match_index]['obj_track_embedding'].append(
                                D[frame_index]['rest_track_embeddings'][
                                    instance_match_index].detach().cpu())  # e.g. embedding
                            Ta[trajectory_match_index]['sha_track_embedding'].append(
                                Ta[trajectory_match_index]['sha_track_embedding'][-1])
                            Ta[trajectory_match_index]['obj_mask'].append(
                                D[frame_index]['rest_masks'][instance_match_index])
                            Ta[trajectory_match_index]['sha_mask'].append(
                                torch.zeros(D[frame_index]['rest_masks'][instance_match_index].shape))
                        else:
                            Ta[trajectory_match_index]['sha_track_embedding'].append(
                                D[frame_index]['rest_track_embeddings'][
                                    instance_match_index].detach().cpu())  # e.g. embedding
                            Ta[trajectory_match_index]['obj_track_embedding'].append(
                                Ta[trajectory_match_index]['obj_track_embedding'][-1])
                            Ta[trajectory_match_index]['sha_mask'].append(
                                D[frame_index]['rest_masks'][instance_match_index])
                            Ta[trajectory_match_index]['obj_mask'].append(
                                torch.zeros(D[frame_index]['rest_masks'][instance_match_index].shape))

                        Ta[trajectory_match_index]['asso_track_embedding'].append(
                            np.concatenate((Ta[trajectory_match_index]['obj_track_embedding'][-1],
                                            Ta[trajectory_match_index]['sha_track_embedding'][-1]), 0))  # e.g. embedding

                        Ta[trajectory_match_index]['asso_mask'].append(D[frame_index]['rest_masks'][instance_match_index])

                        Ta[trajectory_match_index]['obj_score'].append(D[frame_index]['rest_scores'][instance_match_index])
                        Ta[trajectory_match_index]['sha_score'].append(D[frame_index]['rest_scores'][instance_match_index])
                        Ta[trajectory_match_index]['asso_score'].append(D[frame_index]['rest_scores'][instance_match_index])

                        Ta[trajectory_match_index]['index'].append(frame_index)  # e.g. 0
                        Ta[trajectory_match_index]['frame_name'].append(D[frame_index]['index'])  # e.g. '00000.jpg'

                        Ta[trajectory_match_index]["used_score"].append(
                            Ta[trajectory_match_index]['asso_score'][-1])  # e.g. confidence score

                        if D[frame_index]['rest_classes'][instance_match_index] == 0:
                            Ta[trajectory_match_index]['used_masks_obj'].append(
                                D[frame_index]['rest_boxes'][instance_match_index])  # e.g. mask
                            Ta[trajectory_match_index]['used_masks_sha'].append(torch.zeros(
                                D[frame_index]['rest_boxes'][instance_match_index].shape))  # e.g. mask
                        else:
                            Ta[trajectory_match_index]['used_masks_sha'].append(
                                D[frame_index]['rest_boxes'][instance_match_index])  # e.g. mask
                            Ta[trajectory_match_index]['used_masks_obj'].append(torch.zeros(
                                D[frame_index]['rest_boxes'][instance_match_index].shape))  # e.g. mask

                        if match_mode == 0:
                            if D[frame_index]['rest_classes'][instance_match_index] == 0:
                                new_use_emb = Ta[trajectory_match_index]['used_track_embedding'][-1] * momentum + \
                                            Ta[trajectory_match_index]['obj_track_embedding'][-1] * (1 - momentum)
                            else:
                                new_use_emb = Ta[trajectory_match_index]['used_track_embedding'][-1]
                        elif match_mode == 1:
                            if D[frame_index]['rest_classes'][instance_match_index] == 1:
                                new_use_emb = Ta[trajectory_match_index]['used_track_embedding'][-1] * momentum + \
                                            Ta[trajectory_match_index]['sha_track_embedding'][-1] * (1 - momentum)
                            else:
                                new_use_emb = Ta[trajectory_match_index]['used_track_embedding'][-1]
                        elif match_mode == 2:
                            new_use_emb = Ta[trajectory_match_index]['used_track_embedding'][-1] * momentum + \
                                        Ta[trajectory_match_index]['asso_track_embedding'][-1] * (1 - momentum)

                        Ta[trajectory_match_index]['used_track_embedding'].append(new_use_emb)
                        Ta_MatchQueueIndexs.append(trajectory_match_index)
                        Rest_frame_instance_matched[-1].append(instance_match_index)
                    #     print("Match Ins, Tra, fra: ", instance_match_index, trajectory_match_index, frame_index)

                    # else:
                    #     print(instance_match_index, trajectory_match_index, "Rest失败 ", "best_score: ",
                    #         rest_score_matrix[instance_match_index][trajectory_match_index][0],
                    #         "best_sim: ", rest_score_matrix[instance_match_index][trajectory_match_index][1],
                    #         "best_confidence_score: ", rest_score_matrix[instance_match_index][trajectory_match_index][2],
                    #         "best_iou: ", rest_score_matrix[instance_match_index][trajectory_match_index][3])
                    #     print("TQ Has Been Taken: ", trajectory_match_index in Ta_MatchQueueIndexs)
                    #     print("-" * 100)
            # '''
            #########################################################

            # The last unmatched Instance will be used as the first frame of the new tracking queue.
            for instance_index, instance_mask in enumerate(D[frame_index]["used_masks_obj"]):
                # 判断该Instance是否被使用过
                if instance_index in used_instance_index:
                    continue
                new_t_i = dict()
                new_t_i['index'] = []
                new_t_i['frame_name'] = []
                new_t_i['used_score'] = []
                new_t_i['used_masks_obj'] = []
                new_t_i['used_masks_sha'] = []
                new_t_i['obj_mask'] = []
                new_t_i['sha_mask'] = []
                new_t_i['asso_mask'] = []

                new_t_i['obj_track_embedding'] = []
                new_t_i['sha_track_embedding'] = []
                new_t_i['asso_track_embedding'] = []
                new_t_i['used_track_embedding'] = []

                new_t_i['obj_score'] = []
                new_t_i['sha_score'] = []
                new_t_i['asso_score'] = []

                new_t_i['index'].append(frame_index)  # e.g. 0
                new_t_i['frame_name'].append(D[frame_index]['index'])  # e.g. '00000.jpg'
                new_t_i['used_score'].append(D[frame_index]['used_scores'][instance_index])  # e.g. confidence score
                new_t_i['used_masks_obj'].append(D[frame_index]['used_masks_obj'][instance_index])  # e.g. mask
                new_t_i['used_masks_sha'].append(D[frame_index]['used_masks_sha'][instance_index])  # e.g. mask
                new_t_i['used_track_embedding'].append(
                    D[frame_index]['used_track_embeddings'][instance_index])  # e.g. track

                new_t_i['obj_mask'].append(D[frame_index]['obj_masks'][instance_index])
                new_t_i['sha_mask'].append(D[frame_index]['sha_masks'][instance_index])
                new_t_i['asso_mask'].append(D[frame_index]['asso_masks'][instance_index])

                new_t_i['obj_track_embedding'].append(D[frame_index]['obj_track_embeddings'][instance_index])
                new_t_i['sha_track_embedding'].append(D[frame_index]['sha_track_embeddings'][instance_index])
                new_t_i['asso_track_embedding'].append(D[frame_index]['asso_track_embeddings'][instance_index])

                new_t_i['obj_score'].append(D[frame_index]['obj_scores'][instance_index])
                new_t_i['sha_score'].append(D[frame_index]['sha_scores'][instance_index])
                new_t_i['asso_score'].append(D[frame_index]['asso_scores'][instance_index])

                new_Ta.append(new_t_i)
                Ta_lastMatch.append(0)
                Ta_MatchMode[len(Ta_lastMatch) - 1] = match_mode

            # # After a frame is passed, update the Ta Tf queue once
            # for t_i_index in range(len(Ta), len(Ta)+len(new_Ta)):
            #     Ta_MatchMode[t_i_index] = match_mode

            Ta.extend(new_Ta)
            for index_ta, track_active in enumerate(Ta):
                if index_ta not in Ta_MatchQueueIndexs:
                    Ta_lastMatch[index_ta] += 1
                else:
                    Ta_lastMatch[index_ta] = 0

        # End
        for t_i_index in range(len(Ta)):
            # Determine whether the active track has ended

            flag_highest_score = 0
            for instance_socre_info in Ta[t_i_index]['used_score']:
                if instance_socre_info >= sigma_high:
                    flag_highest_score = 1

            if flag_highest_score == 1 and len(Ta[t_i_index]['index']) >= t_min:
                Tf.append(Ta[t_i_index])

        # 20221018: For backward tracking
        ###########################################################
        Ta = Tf
        # Ta_lastMatch = [0 for x in Ta_lastMatch]
        Ta_lastMatch = []  # Assuming Tf contains the final trajectories

        later_Ta_frame_index = 0
        for Ta_each in Ta:
            if Ta_each['index'][0] > later_Ta_frame_index:
                later_Ta_frame_index = Ta_each['index'][0]
        
        for Ta_each in Ta:
            Ta_lastMatch.append(later_Ta_frame_index-Ta_each['index'][0]-1)

        if backward_track == True:
            for frame_index in range(later_Ta_frame_index-1, -1, -1): # the last image cannot be matched for sure, thus -2
                print("Backward Frame: ", frame_index)
                Ta_MatchQueueIndexs = []  # The time of the last match for each trajectory
                if find_missing == True:
                    D[frame_index]['backward_track_rest_used_sim_score_sotfmax_instance'] = list()
                    D[frame_index]['backward_track_rest_used_sim_score_instance'] = list()

                    len_rest = len(D[frame_index]['rest_track_embeddings'])
                    temp_rest_all_tracking_embedding = D[frame_index][
                        'rest_track_embeddings'].detach().cpu().numpy().tolist()
                    temp_rest_all_tracking_embedding.extend(D[frame_index]['used_track_embeddings'])
                    temp_rest_all_mask_obj = D[frame_index]['rest_boxes'].tolist()
                    temp_rest_all_mask_obj.extend(D[frame_index]['used_masks_obj'])
                    temp_rest_all_mask_sha = D[frame_index]['rest_boxes'].tolist()
                    temp_rest_all_mask_sha.extend(D[frame_index]['used_masks_sha'])
                    temp_rest_all_score = D[frame_index]['rest_scores'].detach().cpu().numpy().tolist()
                    temp_rest_all_score.extend(D[frame_index]['used_scores'])
                    if len(Ta) != 0:
                        for instance_index, instance_mask in enumerate(temp_rest_all_tracking_embedding):
                            sim_vector = []
                            for t_i_index in range(len(Ta)):
                                if instance_index < len_rest:
                                    if D[frame_index]["rest_classes"][instance_index] == 0:
                                        t_i_latest_track_embedding = Ta[t_i_index]["obj_track_embedding"][0]
                                    elif D[frame_index]["rest_classes"][instance_index] == 1:
                                        t_i_latest_track_embedding = Ta[t_i_index]["sha_track_embedding"][0]
                                else:
                                    t_i_latest_track_embedding = Ta[t_i_index]["used_track_embedding"][0]
                                temp_track_embedding = temp_rest_all_tracking_embedding[instance_index]
                                if cos_sim == 0:
                                    temp_sim = np.matmul(np.reshape(t_i_latest_track_embedding, (1, -1)),
                                                        np.reshape(temp_track_embedding, (-1, 1)))
                                elif cos_sim == 1:
                                    # Harry New
                                    A = np.reshape(t_i_latest_track_embedding, (1, -1))
                                    B = np.reshape(temp_track_embedding, (-1, 1))
                                    temp_sim = np.dot(A, B) / (norm(A) * norm(B))
                                sim_vector.append(temp_sim.item())
                            sim_vector = np.array(sim_vector)
                            sim_vector_softmax = softmax(sim_vector)

                            D[frame_index]['backward_track_rest_used_sim_score_sotfmax_instance'].append(
                                list(sim_vector_softmax))
                            D[frame_index]['backward_track_rest_used_sim_score_instance'].append(list(sim_vector))
                    else:
                        for instance_index, instance_mask in enumerate(temp_rest_all_tracking_embedding):
                            D[frame_index]['backward_track_rest_used_sim_score_sotfmax_instance'].append([])
                            D[frame_index]['backward_track_rest_used_sim_score_instance'].append([])

                    if len(temp_rest_all_tracking_embedding) != 0:
                        for t_i_index in range(len(Ta)):
                            Ta[t_i_index]['backward_track_rest_used_sim_score_sotfmax_trajectory'] = list()
                            sim_vector = []
                            for instance_index, instance_mask in enumerate(temp_rest_all_tracking_embedding):
                                if instance_index < len_rest:
                                    if D[frame_index]["rest_classes"][instance_index] == 0:
                                        t_i_latest_track_embedding = Ta[t_i_index]["obj_track_embedding"][0]
                                    elif D[frame_index]["rest_classes"][instance_index] == 1:
                                        t_i_latest_track_embedding = Ta[t_i_index]["sha_track_embedding"][0]
                                else:
                                    t_i_latest_track_embedding = Ta[t_i_index]["used_track_embedding"][0]
                                temp_track_embedding = temp_rest_all_tracking_embedding[instance_index]

                                if cos_sim == 0:
                                    temp_sim = np.matmul(np.reshape(t_i_latest_track_embedding, (1, -1)),
                                                        np.reshape(temp_track_embedding, (-1, 1)))
                                elif cos_sim == 1:
                                    # Harry New
                                    A = np.reshape(t_i_latest_track_embedding, (1, -1))
                                    B = np.reshape(temp_track_embedding, (-1, 1))
                                    temp_sim = np.dot(A, B) / (norm(A) * norm(B))
                                sim_vector.append(temp_sim.item())
                            sim_vector = np.array(sim_vector)
                            sim_vector_softmax = softmax(sim_vector)

                            Ta[t_i_index]['backward_track_rest_used_sim_score_sotfmax_trajectory'].extend(
                                list(sim_vector_softmax))
                    else:
                        for t_i_index in range(len(Ta)):
                            Ta[t_i_index]['backward_track_rest_used_sim_score_sotfmax_trajectory'] = list()
                            Ta[t_i_index]['backward_track_rest_used_sim_score_sotfmax_trajectory'].extend([])

                    rest_cost_matrix = []
                    rest_score_matrix = []
                    rest_sim_matrix = []
                    rest_sim_dir_matrix = []
                    for instance_index, instance_mask in enumerate(temp_rest_all_mask_obj):
                        cost_matrix_row = []
                        score_matrix_row = []
                        sim_matrix_row = []
                        sim_dir_matrix_row = []
                        for t_i_index in range(len(Ta)):
                            t_i_latest_mask_obj = Ta[t_i_index]["used_masks_obj"][0]
                            t_i_latest_mask_sha = Ta[t_i_index]["used_masks_sha"][0]
                            last_match_gap = Ta_lastMatch[t_i_index]  # if Ta_lastMatch[t_i_index] < 3 else 3
                            # print("Harry: ", t_i_index, last_match_gap)
                            if instance_index < len_rest:
                                temp_iou_obj, _ = self.compute_ious(torch.tensor(t_i_latest_mask_obj).reshape(1, -1),
                                                            torch.tensor(temp_rest_all_mask_obj[instance_index]).reshape(
                                                                1, -1))
                                temp_iou_obj *= pow(iou_decay_rate, abs(last_match_gap) )
                                temp_iou_sha, _ = self.compute_ious(torch.tensor(t_i_latest_mask_sha).reshape(1, -1),
                                                            torch.tensor(temp_rest_all_mask_sha[instance_index]).reshape(
                                                                1, -1))
                                temp_iou_sha *= pow(iou_decay_rate, abs(last_match_gap))
                                if D[frame_index]["rest_classes"][instance_index] == 0:
                                    temp_iou = temp_iou_obj
                                else:
                                    temp_iou = temp_iou_sha
                            else:
                                temp_iou_obj, _ = self.compute_ious(torch.tensor(t_i_latest_mask_obj).reshape(1, -1),
                                                            torch.tensor(temp_rest_all_mask_obj[instance_index]).reshape(1, -1))
                                temp_iou_obj *= pow(iou_decay_rate, abs(last_match_gap))
                                temp_iou_sha, _ = self.compute_ious(torch.tensor(t_i_latest_mask_sha).reshape(1, -1),
                                                            torch.tensor(temp_rest_all_mask_sha[instance_index]).reshape(1, -1))
                                temp_iou_sha *= pow(iou_decay_rate, abs(last_match_gap))
                                if match_mode == 2:
                                    temp_iou = (temp_iou_obj + temp_iou_sha) * 0.5
                                elif match_mode == 0:
                                    temp_iou = temp_iou_obj
                                else:
                                    temp_iou = temp_iou_sha

                            temp_confidence_score = temp_rest_all_score[instance_index]

                            temp_sim = (D[frame_index]['backward_track_rest_used_sim_score_sotfmax_instance'][
                                            instance_index][t_i_index] +
                                        Ta[t_i_index]['backward_track_rest_used_sim_score_sotfmax_trajectory'][
                                            instance_index]) / 2
                            temp_sim = temp_sim * pow(gap_decay_rate, abs(last_match_gap))
                            temp_score = temp_sim.item() + 0.5 * temp_iou.item() + 0.25 * temp_confidence_score
                            temp_sim_dir = temp_sim.item() + temp_iou.item()
                            score_matrix_row.append([temp_score, temp_sim, temp_confidence_score, temp_iou])
                            if torch.is_tensor(temp_score):
                                temp_score = temp_score.item()
                            cost_matrix_row.append(temp_score)
                            sim_matrix_row.append(temp_sim)
                            sim_dir_matrix_row.append(temp_sim_dir)

                        rest_cost_matrix.append(cost_matrix_row)
                        rest_score_matrix.append(score_matrix_row)
                        rest_sim_matrix.append(sim_matrix_row)
                        rest_sim_dir_matrix.append(sim_dir_matrix_row)
                    if len(temp_rest_all_tracking_embedding) == 0:
                        rest_cost_matrix.append([])
                        rest_sim_matrix.append([])
                        rest_sim_dir_matrix.append([])

                    rest_row_ind, rest_col_ind = linear_sum_assignment(rest_sim_dir_matrix, maximize=True)
                    if len(temp_rest_all_tracking_embedding) > 0 and len(Ta) > 0:
                        rest_sigma_score_final = sigma_score + 1 / (len(Ta)+1) * sigma_score_scale
                                # (1/len(Ta)*sigma_score_scale
                    else:
                        rest_sigma_score_final = sigma_score  # sigma_score_scale

                    for match_index in range(len(rest_row_ind)):
                        instance_match_index = rest_row_ind[match_index]
                        trajectory_match_index = rest_col_ind[match_index]
                        # print(frame_index, Ta[trajectory_match_index]['index'][0])
                        if trajectory_match_index not in Ta_MatchQueueIndexs and \
                                instance_match_index not in Rest_frame_instance_matched[frame_index] and\
                                rest_cost_matrix[instance_match_index][trajectory_match_index] > rest_sigma_score_final and \
                                instance_match_index < len_rest and \
                                frame_index < Ta[trajectory_match_index]['index'][0]:

                            Ta_MatchMode[trajectory_match_index] = D[frame_index]['rest_classes'][instance_match_index]

                            if D[frame_index]['rest_classes'][instance_match_index] == 0:
                                Ta[trajectory_match_index]['obj_track_embedding'].insert(0,
                                                                                        D[frame_index][
                                                                                            'rest_track_embeddings'][
                                                                                            instance_match_index].detach().cpu())  # e.g. embedding
                                Ta[trajectory_match_index]['sha_track_embedding'].insert(0,
                                                                                        Ta[trajectory_match_index][
                                                                                            'sha_track_embedding'][0])
                                Ta[trajectory_match_index]['obj_mask'].insert(0,
                                                                            D[frame_index]['rest_masks'][
                                                                                instance_match_index])
                                Ta[trajectory_match_index]['sha_mask'].insert(0,
                                                                            torch.zeros(D[frame_index]['rest_masks'][
                                                                                            instance_match_index].shape))
                            else:
                                Ta[trajectory_match_index]['sha_track_embedding'].insert(0,
                                                                                        D[frame_index][
                                                                                            'rest_track_embeddings'][
                                                                                            instance_match_index].detach().cpu())  # e.g. embedding
                                Ta[trajectory_match_index]['obj_track_embedding'].insert(0,
                                                                                        Ta[trajectory_match_index][
                                                                                            'obj_track_embedding'][0])
                                Ta[trajectory_match_index]['sha_mask'].insert(0,
                                                                            D[frame_index]['rest_masks'][
                                                                                instance_match_index])
                                Ta[trajectory_match_index]['obj_mask'].insert(0,
                                                                            torch.zeros(D[frame_index]['rest_masks'][
                                                                                            instance_match_index].shape))

                            Ta[trajectory_match_index]['asso_track_embedding'].insert(0,
                                                                                    np.concatenate((Ta[
                                                                                                        trajectory_match_index][
                                                                                                        'obj_track_embedding'][
                                                                                                        0],
                                                                                                    Ta[
                                                                                                        trajectory_match_index][
                                                                                                        'sha_track_embedding'][
                                                                                                        0]),
                                                                                                    0))  # e.g. embedding

                            Ta[trajectory_match_index]['asso_mask'].insert(0,
                                                                        D[frame_index]['rest_masks'][
                                                                            instance_match_index])
                            Ta[trajectory_match_index]['obj_score'].insert(0,
                                                                        D[frame_index]['rest_scores'][
                                                                            instance_match_index])
                            Ta[trajectory_match_index]['sha_score'].insert(0,
                                                                        D[frame_index]['rest_scores'][
                                                                            instance_match_index])
                            Ta[trajectory_match_index]['asso_score'].insert(0,
                                                                            D[frame_index]['rest_scores'][
                                                                                instance_match_index])

                            Ta[trajectory_match_index]['index'].insert(0, frame_index)  # e.g. 0
                            Ta[trajectory_match_index]['frame_name'].insert(0, D[frame_index]['index'])  # e.g. '00000.jpg'

                            Ta[trajectory_match_index]["used_score"].insert(0,
                                                                            Ta[trajectory_match_index]['asso_score'][
                                                                                0])  # e.g. confidence score

                            if D[frame_index]['rest_classes'][instance_match_index] == 0:
                                Ta[trajectory_match_index]['used_masks_obj'].insert(0,
                                                                                    D[frame_index]['rest_boxes'][
                                                                                        instance_match_index])  # e.g. mask
                                Ta[trajectory_match_index]['used_masks_sha'].insert(0, torch.zeros(
                                    D[frame_index]['rest_boxes'][instance_match_index].shape))  # e.g. mask
                            else:
                                Ta[trajectory_match_index]['used_masks_sha'].insert(0,
                                                                                    D[frame_index]['rest_boxes'][
                                                                                        instance_match_index])  # e.g. mask
                                Ta[trajectory_match_index]['used_masks_obj'].insert(0, torch.zeros(
                                    D[frame_index]['rest_boxes'][instance_match_index].shape))  # e.g. mask

                            if match_mode == 0:
                                if D[frame_index]['rest_classes'][instance_match_index] == 0:
                                    new_use_emb = Ta[trajectory_match_index]['used_track_embedding'][0] * momentum + \
                                                Ta[trajectory_match_index]['obj_track_embedding'][0] * (1 - momentum)
                                else:
                                    new_use_emb = Ta[trajectory_match_index]['used_track_embedding'][0]
                            elif match_mode == 1:
                                if D[frame_index]['rest_classes'][instance_match_index] == 1:
                                    new_use_emb = Ta[trajectory_match_index]['used_track_embedding'][0] * momentum + \
                                                Ta[trajectory_match_index]['sha_track_embedding'][0] * (1 - momentum)
                                else:
                                    new_use_emb = Ta[trajectory_match_index]['used_track_embedding'][0]
                            elif match_mode == 2:
                                new_use_emb = Ta[trajectory_match_index]['used_track_embedding'][0] * momentum + \
                                            Ta[trajectory_match_index]['asso_track_embedding'][0] * (1 - momentum)

                            Ta[trajectory_match_index]['used_track_embedding'].insert(0, new_use_emb)
                            Ta_MatchQueueIndexs.append(trajectory_match_index)
                            print("Match Ins, Tra, fra: ", instance_match_index, trajectory_match_index, frame_index)

                for index_ta, track_active in enumerate(Ta):
                    if index_ta not in Ta_MatchQueueIndexs: # and track_active['index'][0] <= frame_index:
                        Ta_lastMatch[index_ta] -= 1
                    else:
                        Ta_lastMatch[index_ta] = 0
        ##############################################

        return Tf


    def pred_function(self, input_path, output_path):
        sigma_low = self.sigma_low
        sigma_high = self.sigma_high
        sigma_score = self.sigma_score
        t_min = self.t_min
        sigma_score_scale = self.sigma_score_scale
        momentum = self.momentum
        cos_sim = self.cos_sim
        iou_decay_rate = self.iou_decay_rate
        gap_decay_rate = self.gap_decay_rate
        iou_nms_th = self.iou_nms_th
        track_mode = self.track_mode
        find_missing = self.find_missing
        backward_track = self.backward_track

        pickle_file = []
        pickle_file.append([])
        pickle_file.append([])

        store_ap = dict()
        store_ap['association'] = dict()
        store_ap['object'] = dict()
        store_ap['shadow'] = dict()
        store_ap['soap'] = dict()

        for IoU_thrshold in [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1]:
            store_ap['association'][str(IoU_thrshold)] = dict()
            store_ap['object'][str(IoU_thrshold)] = dict()
            store_ap['shadow'][str(IoU_thrshold)] = dict()
            store_ap['soap'][str(IoU_thrshold)] = dict()

            for data_type in ['association', 'object', 'shadow', 'soap']:
                store_ap[data_type][str(IoU_thrshold)]['true_positive_list'] = []
                store_ap[data_type][str(IoU_thrshold)]['false_positive_list'] = []
                store_ap[data_type][str(IoU_thrshold)]['false_negative_list'] = []
                store_ap[data_type][str(IoU_thrshold)]['total_pd_number'] = 0
                store_ap[data_type][str(IoU_thrshold)]['total_gd_number'] = 0

        if os.path.exists(output_path):
            os.system("rm -rf "+output_path)
        os.mkdir(output_path)

        os.mkdir(output_path+"/all_masks")
        os.mkdir(output_path+"/painted_image")

        # Automatically detect input type
        if isinstance(input_path, list):
            # If input_path is already a list (e.g., frames from a video), use it directly
            image_paths = input_path
        elif isinstance(input_path, str):
            if os.path.isdir(input_path):
                # Directory of images
                image_paths = sorted(glob.glob(os.path.join(input_path, "*")), key=extract_number)
            elif os.path.isfile(input_path):
                if input_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                    # Single image file
                    image_paths = [input_path]
                elif input_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                    # Video file: extract frames
                    image_paths = self.extract_frames_from_video(input_path)
                else:
                    raise ValueError("Unsupported file format. Provide an image, directory, or video file.")
            else:
                raise FileNotFoundError(f"Input path '{input_path}' does not exist.")
        else:
            raise TypeError("Input path must be a string, list, or valid path-like object.")

        input_path = "./tmp/video_frames/"


        # Ensure output path exists
        if not os.path.exists(output_path):
            os.makedirs(output_path)



        # path_ssis预测结果
        result_pred = self.pred_result_generate(input_path, output_path)
        for frame_index in range(len(result_pred)):
            # save result_pred[frame_index]['pred_masks;] and result_pred[frame_index]['rest_pred_masks']
            os.mkdir(output_path+"/all_masks/"+str(frame_index))
            for mask_id, pred_mask in enumerate(result_pred[frame_index]['pred_masks']):
                if not isinstance(pred_mask, np.ndarray):
                    temp_mask = pred_mask.cpu().numpy()
                    im = Image.fromarray((temp_mask * 255).astype(np.uint8))
                    if im.mode == "F":
                        im = im.convert('RGB')
                    im.save(output_path+"/all_masks/"+str(frame_index)+"/pred_mask_"+str(mask_id)+".png")
            # same for rest_pred_masks
            for mask_id, rest_pred_mask in enumerate(result_pred[frame_index]['rest_pred_masks']):
                if not isinstance(rest_pred_mask, np.ndarray):
                    temp_mask = rest_pred_mask.cpu().numpy()
                    im = Image.fromarray((temp_mask * 255).astype(np.uint8))
                    if im.mode == "F":
                        im = im.convert('RGB')
                    im.save(output_path+"/all_masks/"+str(frame_index)+"/rest_pred_mask_"+str(mask_id)+".png")
                    

        # 数据预处理: 获取Object, Shadow, Association掩码
        total_instance_pred = 0
        for frame_index in range(len(result_pred)):
            obj_masks = []
            obj_boxes = []
            sha_masks = []
            sha_boxes = []
            asso_masks = []

            obj_scores = []
            sha_scores = []
            asso_scores = []

            obj_track_embeddings = []
            sha_track_embeddings = []
            asso_track_embeddings = []
            last_class = -1
            class_count = 0
            for pred_class_index in range(len(result_pred[frame_index]['pred_classes'])):
                if result_pred[frame_index]['pred_classes'][pred_class_index] == 0 and last_class != 0:
                    last_class = 0
                    class_count += 1

                    obj_mask = result_pred[frame_index]['pred_masks'][pred_class_index]
                    obj_box = result_pred[frame_index]['pred_boxes'][pred_class_index]
                    obj_score = result_pred[frame_index]['scores'][pred_class_index]
                    obj_track_embedding = result_pred[frame_index]['track_embedding'][pred_class_index]

                    obj_masks.append(obj_mask)
                    obj_boxes.append(obj_box)
                    obj_scores.append(obj_score)
                    obj_track_embeddings.append(obj_track_embedding)

                elif result_pred[frame_index]['pred_classes'][pred_class_index] == 1 and last_class != 1:
                    last_class = 1
                    class_count += 1

                    sha_mask = result_pred[frame_index]['pred_masks'][pred_class_index]
                    sha_box = result_pred[frame_index]['pred_boxes'][pred_class_index]
                    sha_score = result_pred[frame_index]['scores'][pred_class_index]
                    sha_track_embedding = result_pred[frame_index]['track_embedding'][pred_class_index]

                    sha_masks.append(sha_mask)
                    sha_boxes.append(sha_box)
                    sha_scores.append(sha_score)
                    sha_track_embeddings.append(sha_track_embedding)

                if class_count % 2 == 0:
                    last_class = -1
                    class_count = 0
                    if sha_mask[obj_mask > 0].sum() == 0:  # combine object and shadow mask
                        asso_mask = obj_mask + sha_mask
                    else:
                        # print(i,shadow_mask[object_mask>0].sum() )

                        sha_mask[obj_mask > 0] = 0
                        asso_mask = obj_mask + sha_mask
                    asso_score = (obj_score + sha_score) / 2
                    asso_track_embedding = np.concatenate((obj_track_embedding, sha_track_embedding), 0)

                    asso_masks.append(asso_mask)
                    asso_scores.append(asso_score)
                    asso_track_embeddings.append(asso_track_embedding)

                    total_instance_pred += len(asso_scores)

            # print("Legngth: ", len(obj_masks), len(sha_masks), len(asso_masks))
            maximum_number = min(len(obj_masks), len(sha_masks), len(asso_masks))
            result_pred[frame_index]['obj_masks'] = obj_masks[:maximum_number]
            result_pred[frame_index]['obj_boxes'] = obj_boxes[:maximum_number]
            result_pred[frame_index]['sha_masks'] = sha_masks[:maximum_number]
            result_pred[frame_index]['sha_boxes'] = sha_boxes[:maximum_number]
            result_pred[frame_index]['asso_masks'] = asso_masks[:maximum_number]

            result_pred[frame_index]['obj_scores'] = obj_scores[:maximum_number]
            result_pred[frame_index]['sha_scores'] = sha_scores[:maximum_number]
            result_pred[frame_index]['asso_scores'] = asso_scores[:maximum_number]

            result_pred[frame_index]['obj_track_embeddings'] = obj_track_embeddings[:maximum_number]
            result_pred[frame_index]['sha_track_embeddings'] = sha_track_embeddings[:maximum_number]
            result_pred[frame_index]['asso_track_embeddings'] = asso_track_embeddings[:maximum_number]

            # New 20220724

            del_index_list = []
            used_confidence_index = []
            asso_scores_list = [x for x in result_pred[frame_index]['asso_scores']]
            confidence_rank_list = np.argsort(asso_scores_list)[::-1]

            for confidence_index0 in confidence_rank_list:
                high_confidence_asso_mask = result_pred[frame_index]['asso_masks'][confidence_index0]
                high_confidence_obj_mask = result_pred[frame_index]['obj_masks'][confidence_index0]
                high_confidence_sha_mask = result_pred[frame_index]['sha_masks'][confidence_index0]
                for confidence_index1 in confidence_rank_list:
                    if confidence_index1 == confidence_index0 or confidence_index1 in used_confidence_index: continue
                    low_confidence_asso_mask = result_pred[frame_index]['asso_masks'][confidence_index1]
                    low_confidence_obj_mask = result_pred[frame_index]['obj_masks'][confidence_index1]
                    low_confidence_sha_mask = result_pred[frame_index]['sha_masks'][confidence_index1]
                    iou_high_low_asso = self.binaryMaskIOU(high_confidence_asso_mask, low_confidence_asso_mask)
                    # iou_high_low_obj = self.binaryMaskIOU(high_confidence_obj_mask, low_confidence_obj_mask)
                    # iou_high_low_sha = self.binaryMaskIOU(high_confidence_sha_mask, low_confidence_sha_mask)
                    if iou_high_low_asso > iou_nms_th:
                        used_confidence_index.append(confidence_index1)
                        if confidence_index1 not in del_index_list:
                            del_index_list.append(confidence_index1)
                    else:
                        result_pred[frame_index]['obj_masks'][confidence_index1][
                            result_pred[frame_index]['obj_masks'][confidence_index0] > 0] = 0
                        result_pred[frame_index]['sha_masks'][confidence_index1][
                            result_pred[frame_index]['sha_masks'][confidence_index0] > 0] = 0
                        result_pred[frame_index]['asso_masks'][confidence_index1][
                            result_pred[frame_index]['asso_masks'][confidence_index0] > 0] = 0
                used_confidence_index.append(confidence_index0)
            del_index_list.sort()
            del_index_list.reverse()
            for result_pred_key in result_pred[frame_index].keys():
                if type(result_pred[frame_index][result_pred_key]) is not list: continue
                for del_index in del_index_list:
                    result_pred[frame_index][result_pred_key].pop(del_index)
            total_instance_pred -= len(del_index_list)

            # 20221002 for missing instances
            if find_missing == True:
                # Rest and Paired IoU
                ori_rest_classes = result_pred[frame_index]['rest_pred_classes'].detach().cpu().numpy()
                ori_rest_scores = result_pred[frame_index]['rest_scores'].detach().cpu().numpy()
                keep_indices = [a and b for a, b in zip(ori_rest_classes!=3, ori_rest_scores>0.0)]
                result_pred[frame_index]['rest_track_embeddings'] = result_pred[frame_index]['rest_track_embeddings'][keep_indices]
                result_pred[frame_index]['rest_masks'] = result_pred[frame_index]['rest_pred_masks'][keep_indices]
                result_pred[frame_index]['rest_asso_masks'] = result_pred[frame_index]['rest_pred_asso_masks'][keep_indices]
                result_pred[frame_index]['rest_scores'] = result_pred[frame_index]['rest_scores'][keep_indices]
                result_pred[frame_index]['rest_classes'] = result_pred[frame_index]['rest_pred_classes'][keep_indices]
                result_pred[frame_index]['rest_boxes'] = result_pred[frame_index]['rest_pred_boxes'][keep_indices]
                rest_del_index_list = []
                for rest_index in range(len(result_pred[frame_index]['rest_boxes'])):
                    for main_index in range(len(result_pred[frame_index]['obj_masks'])):
                        if result_pred[frame_index]['rest_classes'][rest_index] == 0:
                            # iou_rest_main, _ = self.compute_ious(result_pred[frame_index]['rest_boxes'][rest_index].reshape(1,-1),result_pred[frame_index]['obj_boxes'][main_index].reshape(1,-1))
                            iou_rest_main = self.binaryMaskIOU(result_pred[frame_index]['rest_masks'][rest_index], result_pred[frame_index]['obj_masks'][main_index])

                        else:
                            # iou_rest_main, _ = self.compute_ious(result_pred[frame_index]['rest_boxes'][rest_index].reshape(1,-1),result_pred[frame_index]['sha_boxes'][main_index].reshape(1,-1))
                            iou_rest_main = self.binaryMaskIOU(result_pred[frame_index]['rest_masks'][rest_index], result_pred[frame_index]['sha_masks'][main_index])

                        if iou_rest_main > iou_nms_th:
                            if rest_index not in rest_del_index_list:
                                rest_del_index_list.append(rest_index)
                        else:
                            result_pred[frame_index]['rest_masks'][rest_index][result_pred[frame_index]['asso_masks'][main_index] > 0] = 0
                            result_pred[frame_index]['rest_asso_masks'][rest_index][result_pred[frame_index]['asso_masks'][main_index] > 0] = 0
                            if np.count_nonzero(result_pred[frame_index]['rest_masks'][rest_index]) == 0:
                                rest_del_index_list.append(rest_index)
                rest_del_index_list.sort()
                rest_del_index_list.reverse()
                for del_index in rest_del_index_list:
                    for result_pred_key in ['rest_track_embeddings','rest_masks','rest_asso_masks','rest_boxes','rest_scores','rest_classes']:
                        result_pred[frame_index][result_pred_key] = torch.cat([result_pred[frame_index][result_pred_key][0:del_index], result_pred[frame_index][result_pred_key][del_index+1:]])

            #########
        print("total_instance_pred: ", total_instance_pred)

        # IoU Tracker部分
        ################################################
        Tf = self.SSIS_Tracker(result_pred, track_mode, sigma_low, sigma_high, sigma_score, t_min, sigma_score_scale,
                        momentum, cos_sim, iou_decay_rate, gap_decay_rate, find_missing, backward_track)

        temp = [input_path + "/*"]
        paths = sorted(glob.glob(temp[0]), key=extract_number)

        # 得到原视频的每一个Frame
        vid_frames = []
        for path in paths:
            temp = dict()
            # img = cv2.imread(path)  # format="BGR" JPG
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED) # PNG
            temp["index"] = path.split("/")[-1]
            temp["image"] = img
            vid_frames.append(temp)

        all_traj_masks = []
        for _ in Tf:
            all_kinds_masks = dict()
            all_kinds_masks["obj_mask"] = []
            all_kinds_masks["sha_mask"] = []
            all_kinds_masks["asso_mask"] = []
            all_traj_masks.append(all_kinds_masks)

        '''
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        size = vid_frames[0]['image'].shape
        out = cv2.VideoWriter(("./test.mp4"), fourcc, 10.0,
                            (size[1], size[0]), True)
        '''

        for pred_tra_index in range(len(Tf)):
            os.mkdir(output_path+"/instance_"+str(pred_tra_index))
            os.mkdir(output_path+"/instance_"+str(pred_tra_index)+"/obj_mask")
            os.mkdir(output_path+"/instance_"+str(pred_tra_index)+"/sha_mask")
            os.mkdir(output_path+"/instance_"+str(pred_tra_index)+"/asso_mask")

            os.mkdir(output_path+"/instance_"+str(pred_tra_index)+"/obj_matte")
            os.mkdir(output_path+"/instance_"+str(pred_tra_index)+"/sha_matte")
            os.mkdir(output_path+"/instance_"+str(pred_tra_index)+"/asso_matte")
            os.mkdir(output_path+"/instance_"+str(pred_tra_index)+"/painted_image")
        
        for frame_id, vid_frame in enumerate(vid_frames):
            frame = vid_frame['image']
            frame_painted = frame

            for pred_tra_index in range(len(Tf)):
                # Hsing
                if vid_frame['index'] in Tf[pred_tra_index]["frame_name"]:
                    mask_index = Tf[pred_tra_index]["frame_name"].index(vid_frame['index'])
                    asso_mask = Tf[pred_tra_index]["asso_mask"][mask_index]
                    obj_mask = Tf[pred_tra_index]["obj_mask"][mask_index]
                    # obj_mask = np.stack((obj_mask, obj_mask, obj_mask),axis=2)
                    sha_mask = Tf[pred_tra_index]["sha_mask"][mask_index]
                    # sha_mask = np.stack((sha_mask, sha_mask, sha_mask),axis=2)
                    if not isinstance(obj_mask, np.ndarray):
                        obj_mask = obj_mask.cpu().numpy()
                        Tf[pred_tra_index]["obj_mask"][mask_index] = obj_mask
                    if not isinstance(sha_mask, np.ndarray):
                        sha_mask = sha_mask.cpu().numpy()
                        Tf[pred_tra_index]["sha_mask"][mask_index] = sha_mask

                    if not isinstance(asso_mask, np.ndarray):
                        asso_mask = asso_mask.cpu().numpy()
                        Tf[pred_tra_index]["asso_mask"][mask_index] = asso_mask

                    frame_painted_ins = mask_painter(frame, (obj_mask).astype('uint8'), mask_color=pred_tra_index+1)
                    frame_painted_ins = mask_painter(frame_painted_ins, (sha_mask).astype('uint8'), mask_color=pred_tra_index+1)

                    frame_painted = mask_painter(frame_painted, (obj_mask).astype('uint8'), mask_color=pred_tra_index+1)
                    frame_painted = mask_painter(frame_painted, (sha_mask).astype('uint8'), mask_color=pred_tra_index+1)

                    # Apply dilation to masks
                    obj_mask = dilate_mask(obj_mask)
                    sha_mask = dilate_mask(sha_mask)
                    asso_mask = dilate_mask(asso_mask)

                    all_traj_masks[pred_tra_index]["obj_mask"].append((obj_mask).astype('uint8'))
                    all_traj_masks[pred_tra_index]["sha_mask"].append((sha_mask).astype('uint8'))
                    all_traj_masks[pred_tra_index]["asso_mask"].append((asso_mask).astype('uint8'))

                    im = Image.fromarray((obj_mask * 255).astype(np.uint8))
                    if im.mode == "F":
                        im = im.convert('RGB')
                    im.save(output_path+"/instance_"+str(pred_tra_index)+"/obj_mask" + "/" + "frame" + str(frame_id) + ".png")
                
                    im = Image.fromarray((sha_mask * 255).astype(np.uint8))
                    if im.mode == "F":
                        im = im.convert('RGB')
                    im.save(output_path+"/instance_"+str(pred_tra_index)+"/sha_mask" + "/" + "frame" + str(frame_id) + ".png")
                    
                    im = Image.fromarray((asso_mask * 255).astype(np.uint8))
                    if im.mode == "F":
                        im = im.convert('RGB')
                    im.save(output_path+"/instance_"+str(pred_tra_index)+"/asso_mask" + "/" + "frame" + str(frame_id) + ".png")

                    frame_painted_ins = cv2.cvtColor(frame_painted_ins, cv2.COLOR_BGR2RGB)
                    im = Image.fromarray((frame_painted_ins).astype(np.uint8))
                    if im.mode == "F":
                        im = im.convert('RGB')
                    im.save(output_path+"/instance_"+str(pred_tra_index)+"/painted_image" + "/" + "frame" + str(frame_id) + ".png")



                    # Process each mask type, tranparent backgound, too time comsuming
                    # for mask_type in ["obj", "sha", "asso"]:
                    #     mask = Tf[pred_tra_index][f"{mask_type}_mask"][mask_index]

                    #     # Convert frame to RGB if it's in BGR
                    #     frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    #     # Create an RGBA image for the matte
                    #     matte = np.zeros((frame.shape[0], frame.shape[1], 4), dtype=np.uint8)  # 4 channels for RGBA
                    #     for c in range(3):  # Copy RGB channels
                    #         matte[:, :, c] = frame_rgb[:, :, c]
                    #     matte[:, :, 3] = mask * 255  # Alpha channel represents the mask

                    #     # Convert the matte to an image and save it
                    #     matte_im = Image.fromarray(matte)
                    #     matte_im.save(output_path + f"/instance_{pred_tra_index}/{mask_type}_matte/" + "frame" + str(frame_id) + ".png")

                    # Process each mask type, black background
                    for mask_type in ["obj", "sha", "asso"]:
                        mask = Tf[pred_tra_index][f"{mask_type}_mask"][mask_index]

                        # Convert frame to RGB if it's in BGR
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                        # Apply the mask to the frame to create a matte
                        matte = np.copy(frame_rgb)
                        for c in range(3):  # Assuming frame is in RGB format
                            matte[:, :, c] = matte[:, :, c] * mask

                        # # Convert back to BGR if needed for saving
                        # matte_bgr = cv2.cvtColor(matte, cv2.COLOR_RGB2BGR)

                        # Convert the matte to an image and save it
                        matte_im = Image.fromarray(matte)  # Save in BGR format if using OpenCV
                        matte_im.save(output_path + f"/instance_{pred_tra_index}/{mask_type}_matte/" + "frame" + str(frame_id) + ".png")

                else:
                    empty_mask = np.zeros((frame.shape[0], frame.shape[1])).astype('uint8')
                    all_traj_masks[pred_tra_index]["obj_mask"].append(empty_mask)
                    all_traj_masks[pred_tra_index]["sha_mask"].append(empty_mask)
                    all_traj_masks[pred_tra_index]["asso_mask"].append(empty_mask)

                    im = Image.fromarray((empty_mask * 255).astype(np.uint8))
                    if im.mode == "F":
                        im = im.convert('RGB')
                    im.save(output_path+"/instance_"+str(pred_tra_index)+"/obj_mask" + "/" + "frame" + str(frame_id) + ".png")
                    im.save(output_path+"/instance_"+str(pred_tra_index)+"/sha_mask" + "/" + "frame" + str(frame_id) + ".png")
                    im.save(output_path+"/instance_"+str(pred_tra_index)+"/asso_mask" + "/" + "frame" + str(frame_id) + ".png")

                    # Save an empty matte image for each mask type
                    for mask_type in ["obj", "sha", "asso"]:
                        empty_matte = np.zeros(frame.shape, dtype=np.uint8)  # Creating an empty matte with the same dimensions as the frame
                        matte_im = Image.fromarray(empty_matte)
                        matte_im.save(output_path + f"/instance_{pred_tra_index}/{mask_type}_matte/" + "frame" + str(frame_id) + ".png")
                        
                    frame_save = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    im = Image.fromarray((frame_save).astype(np.uint8))
                    if im.mode == "F":
                        im = im.convert('RGB')
                    im.save(output_path+"/instance_"+str(pred_tra_index)+"/painted_image" + "/" + "frame" + str(frame_id) + ".png")    

            frame_painted = cv2.cvtColor(frame_painted, cv2.COLOR_BGR2RGB)
            im = Image.fromarray((frame_painted).astype(np.uint8))
            if im.mode == "F":
                im = im.convert('RGB')
            im.save(output_path+"/painted_image"+ "/" + "frame" + str(frame_id) + ".png")
        '''
            out.write(frame_painted)
        out.release()
        #'''

        return all_traj_masks


if __name__ == "__main__":
    print(torch.cuda.is_available())
    args = get_parser().parse_args()

    visd_tracker = VisdTracker()

    visd_tracker.pred_function(args.input_name, args.output_name)

    painted_image_folder = os.path.join(args.output_name, "painted_image")
    output_mp4_path = os.path.join(args.output_name, "output_video.mp4")
    generate_video_from_frames(painted_image_folder, output_mp4_path, frame_rate=10)

