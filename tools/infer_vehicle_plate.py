# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Vehicle Detection + License Plate Recognition Pipeline

This script combines vehicle detection with license plate detection and recognition.
It uses PaddleDetection for vehicle detection and PaddleOCR for plate OCR.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

# add python path of PaddleDetection to sys.path
parent_path = os.path.abspath(os.path.join(__file__, *(['..'] * 2)))
sys.path.insert(0, parent_path)

# ignore warning log
import warnings
warnings.filterwarnings('ignore')

import glob
import ast
import argparse
import cv2
import numpy as np
from paddleocr import PaddleOCR

import paddle
from ppdet.core.workspace import create, load_config, merge_config
from ppdet.engine import Trainer
from ppdet.utils.check import check_gpu, check_npu, check_xpu, check_mlu, check_version, check_config
from ppdet.utils.cli import ArgsParser, merge_args
from ppdet.utils.logger import setup_logger
from ppdet.utils.plate_utils import (
    crop_vehicle_region,
    detect_and_recognize_plates,
    visualize_results,
    print_results
)

logger = setup_logger('vehicle_plate_infer')


def str2bool(v):
    """Convert string to boolean"""
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def parse_args():
    parser = ArgsParser()
    parser.add_argument(
        "--infer_dir",
        type=str,
        default=None,
        help="Directory for images to perform inference on.")
    parser.add_argument(
        "--infer_img",
        type=str,
        default=None,
        help="Image path, has higher priority over --infer_dir")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output",
        help="Directory for storing the output visualization files.")
    parser.add_argument(
        "--draw_threshold",
        type=float,
        default=0.5,
        help="Threshold to reserve the result for visualization.")
    parser.add_argument(
        "--plate_det_threshold",
        type=float,
        default=0.3,
        help="Threshold for plate detection confidence.")
    parser.add_argument(
        "--use_gpu",
        type=str2bool,
        default=False,
        help="Whether to use GPU for vehicle detection.")
    parser.add_argument(
        "--ocr_use_gpu",
        type=str2bool,
        default=False,
        help="Whether to use GPU for OCR.")
    parser.add_argument(
        "--ocr_lang",
        type=str,
        default='ch',
        help="OCR language: 'ch' for Chinese, 'en' for English.")
    args = parser.parse_args()
    return args


def get_test_images(infer_dir, infer_img):
    """
    Get image path list in TEST mode
    """
    assert infer_img is not None or infer_dir is not None, \
        "--infer_img or --infer_dir should be set"
    assert infer_img is None or os.path.isfile(infer_img), \
            "{} is not a file".format(infer_img)
    assert infer_dir is None or os.path.isdir(infer_dir), \
            "{} is not a directory".format(infer_dir)

    # infer_img has a higher priority
    if infer_img and os.path.isfile(infer_img):
        return [infer_img]

    images = set()
    infer_dir = os.path.abspath(infer_dir)
    assert os.path.isdir(infer_dir), \
        "infer_dir {} is not a directory".format(infer_dir)
    
    exts = ['jpg', 'jpeg', 'png', 'bmp']
    exts += [ext.upper() for ext in exts]
    for ext in exts:
        images.update(glob.glob('{}/*.{}'.format(infer_dir, ext)))
    images = list(images)
    assert len(images) > 0, "no image found in {}".format(infer_dir)
    logger.info("Found {} inference images in total.".format(len(images)))

    return images


def detect_vehicles(trainer, image_path, draw_threshold):
    """
    Detect vehicles in an image using PaddleDetection.
    
    Returns:
        image: Original image as numpy array
        results: List of detection results [{'bbox': [x1,y1,x2,y2], 'score': conf, 'class': 'vehicle'}, ...]
    """
    # Read image
    image = cv2.imread(image_path)
    if image is None:
        logger.error(f"Failed to read image: {image_path}")
        return None, []
    
    # Run vehicle detection - returns list of dicts with 'bbox', 'bbox_num', etc.
    results = trainer.predict([image_path], draw_threshold=0.0,  # Get all detections
                             visualize=False, save_results=False)
    
    # Parse results
    vehicle_results = []
    if results and len(results) > 0:
        result = results[0]  # First (and only) image
        if 'bbox' in result and result['bbox'] is not None:
            bboxes = result['bbox']
            # bboxes format: numpy array where each row is [class_id, score, x1, y1, x2, y2]
            for bbox in bboxes:
                if len(bbox) >= 6:
                    class_id, score, x1, y1, x2, y2 = bbox[:6]
                    # Apply threshold filter here
                    if score >= draw_threshold:
                        vehicle_results.append({
                            'bbox': [float(x1), float(y1), float(x2), float(y2)],
                            'score': float(score),
                            'class': int(class_id)
                        })
    
    return image, vehicle_results


def process_image(image_path, trainer, ocr_engine, args):
    """
    Process a single image: detect vehicles and recognize license plates.
    """
    logger.info(f"Processing: {image_path}")
    
    # Step 1: Detect vehicles
    image, vehicle_results = detect_vehicles(trainer, image_path, args.draw_threshold)
    
    if image is None:
        return
    
    if len(vehicle_results) == 0:
        logger.info(f"No vehicles detected in {image_path}")
        return
    
    # Step 2: For each vehicle, detect and recognize license plates
    plate_results_per_vehicle = []
    for vehicle in vehicle_results:
        # Crop vehicle region
        vehicle_crop = crop_vehicle_region(image, vehicle['bbox'])
        
        # Detect and recognize plates in this vehicle
        plate_results = detect_and_recognize_plates(
            ocr_engine, 
            vehicle_crop, 
            det_threshold=args.plate_det_threshold
        )
        
        plate_results_per_vehicle.append(plate_results)
    
    # Step 3: Visualize results
    vis_image = visualize_results(
        image, 
        vehicle_results, 
        plate_results_per_vehicle,
        draw_threshold=args.draw_threshold
    )
    
    # Step 4: Save output
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, os.path.basename(image_path))
    cv2.imwrite(output_path, vis_image)
    logger.info(f"Saved result to: {output_path}")
    
    # Step 5: Print results
    print_results(image_path, vehicle_results, plate_results_per_vehicle)


def run(FLAGS, cfg):
    """
    Main inference pipeline
    """
    # Initialize vehicle detection trainer
    trainer = Trainer(cfg, mode='test')
    trainer.load_weights(cfg.weights)
    
    # Initialize PaddleOCR for plate detection and recognition
    logger.info("Initializing PaddleOCR...")
    ocr_engine = PaddleOCR(
        use_textline_orientation=False,  # Don't need angle classification for plates
        lang=FLAGS.ocr_lang
    )
    logger.info("PaddleOCR initialized successfully")
    
    # Get inference images
    images = get_test_images(FLAGS.infer_dir, FLAGS.infer_img)
    
    # Process each image
    for image_path in images:
        try:
            process_image(image_path, trainer, ocr_engine, FLAGS)
        except Exception as e:
            logger.error(f"Error processing {image_path}: {e}")
            import traceback
            traceback.print_exc()
    
    logger.info(f"Inference complete! Results saved to: {FLAGS.output_dir}")


def main():
    FLAGS = parse_args()
    cfg = load_config(FLAGS.config)
    merge_args(cfg, FLAGS)
    merge_config(FLAGS.opt)

    # Set device
    if 'use_gpu' not in cfg:
        cfg.use_gpu = FLAGS.use_gpu
    
    if cfg.use_gpu:
        place = paddle.set_device('gpu')
    else:
        place = paddle.set_device('cpu')

    check_config(cfg)
    check_gpu(cfg.use_gpu)
    check_version()
    
    run(FLAGS, cfg)


if __name__ == '__main__':
    main()
