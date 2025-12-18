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
Utility functions for license plate detection and recognition.
"""

import cv2
import numpy as np
from typing import List, Tuple, Dict, Any


def crop_vehicle_region(image: np.ndarray, bbox: List[float]) -> np.ndarray:
    """
    Crop vehicle region from the full image based on bounding box.
    
    Args:
        image: Full image as numpy array (H, W, C)
        bbox: Bounding box as [x1, y1, x2, y2] in pixel coordinates
        
    Returns:
        Cropped image region
    """
    x1, y1, x2, y2 = map(int, bbox)
    # Ensure coordinates are within image bounds
    h, w = image.shape[:2]
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w, x2)
    y2 = min(h, y2)
    
    return image[y1:y2, x1:x2]


def detect_and_recognize_plates(
    ocr_engine,
    vehicle_crop: np.ndarray,
    det_threshold: float = 0.3
) -> List[Dict[str, Any]]:
    """
    Detect and recognize license plates in a vehicle crop using PaddleOCR.
    
    Args:
        ocr_engine: PaddleOCR instance
        vehicle_crop: Cropped vehicle image
        det_threshold: Detection confidence threshold
        
    Returns:
        List of dictionaries containing plate detection results:
        [
            {
                'bbox': [[x1,y1], [x2,y2], [x3,y3], [x4,y4]],  # plate bbox in vehicle crop
                'text': 'recognized text',
                'confidence': 0.95
            },
            ...
        ]
    """
    if vehicle_crop is None or vehicle_crop.size == 0:
        return []
    
    try:
        # Run OCR on vehicle crop using predict method
        results = ocr_engine.predict(vehicle_crop)
        
        if results is None or len(results) == 0:
            return []
        
        plate_results = []
        # Results format from predict: list of detection results
        for result in results:
            if result is None:
                continue
            
            # Extract bbox, text, and confidence
            # The result structure may vary, handle different formats
            if isinstance(result, dict):
                bbox = result.get('bbox', result.get('box', None))
                text = result.get('text', result.get('rec_text', ''))
                confidence = result.get('confidence', result.get('rec_score', 0.0))
            elif isinstance(result, (list, tuple)) and len(result) >= 2:
                # Format: (bbox, (text, confidence))
                bbox = result[0]
                if isinstance(result[1], (list, tuple)) and len(result[1]) >= 2:
                    text, confidence = result[1][0], result[1][1]
                else:
                    text = str(result[1])
                    confidence = 0.0
            else:
                continue
            
            # Filter by confidence threshold
            if confidence >= det_threshold:
                plate_results.append({
                    'bbox': bbox,
                    'text': text,
                    'confidence': confidence
                })
        
        return plate_results
    except Exception as e:
        print(f"Error in plate detection: {e}")
        return []


def convert_bbox_to_absolute(
    plate_bbox: List[List[float]],
    vehicle_bbox: List[float]
) -> List[List[float]]:
    """
    Convert plate bounding box from vehicle-relative to image-absolute coordinates.
    
    Args:
        plate_bbox: Plate bbox in vehicle crop coordinates [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
        vehicle_bbox: Vehicle bbox in image coordinates [x1, y1, x2, y2]
        
    Returns:
        Plate bbox in absolute image coordinates
    """
    vx1, vy1 = vehicle_bbox[0], vehicle_bbox[1]
    
    absolute_bbox = []
    for point in plate_bbox:
        abs_x = point[0] + vx1
        abs_y = point[1] + vy1
        absolute_bbox.append([abs_x, abs_y])
    
    return absolute_bbox


def visualize_results(
    image: np.ndarray,
    vehicle_results: List[Dict[str, Any]],
    plate_results_per_vehicle: List[List[Dict[str, Any]]],
    draw_threshold: float = 0.5
) -> np.ndarray:
    """
    Visualize vehicle detection and plate recognition results on the image.
    
    Args:
        image: Original image
        vehicle_results: List of vehicle detection results
            [{'bbox': [x1,y1,x2,y2], 'score': 0.9, 'class': 'vehicle'}, ...]
        plate_results_per_vehicle: List of plate results for each vehicle
            [[plate1, plate2], [plate3], ...]
        draw_threshold: Minimum confidence to draw
        
    Returns:
        Annotated image
    """
    vis_image = image.copy()
    
    for i, vehicle in enumerate(vehicle_results):
        if vehicle['score'] < draw_threshold:
            continue
        
        # Draw vehicle bounding box (green)
        vx1, vy1, vx2, vy2 = map(int, vehicle['bbox'])
        cv2.rectangle(vis_image, (vx1, vy1), (vx2, vy2), (0, 255, 0), 2)
        
        # Draw vehicle label
        label = f"Vehicle: {vehicle['score']:.2f}"
        cv2.putText(vis_image, label, (vx1, vy1 - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Draw plate results for this vehicle
        if i < len(plate_results_per_vehicle):
            plates = plate_results_per_vehicle[i]
            for plate in plates:
                # Convert plate bbox to absolute coordinates
                abs_bbox = convert_bbox_to_absolute(plate['bbox'], vehicle['bbox'])
                
                # Draw plate bounding box (blue)
                pts = np.array(abs_bbox, dtype=np.int32)
                cv2.polylines(vis_image, [pts], True, (255, 0, 0), 2)
                
                # Draw recognized text (red background)
                text = plate['text']
                conf = plate['confidence']
                text_label = f"{text} ({conf:.2f})"
                
                # Get text position (top-left of plate bbox)
                text_x, text_y = int(abs_bbox[0][0]), int(abs_bbox[0][1]) - 10
                
                # Draw text with background
                (text_w, text_h), _ = cv2.getTextSize(
                    text_label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(vis_image, 
                            (text_x, text_y - text_h - 5),
                            (text_x + text_w, text_y + 5),
                            (0, 0, 255), -1)
                cv2.putText(vis_image, text_label, (text_x, text_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    return vis_image


def print_results(
    image_path: str,
    vehicle_results: List[Dict[str, Any]],
    plate_results_per_vehicle: List[List[Dict[str, Any]]]
):
    """
    Print detection and recognition results to console.
    
    Args:
        image_path: Path to the image
        vehicle_results: Vehicle detection results
        plate_results_per_vehicle: Plate results for each vehicle
    """
    print(f"\n{'='*60}")
    print(f"Results for: {image_path}")
    print(f"{'='*60}")
    print(f"Detected {len(vehicle_results)} vehicle(s)")
    
    for i, vehicle in enumerate(vehicle_results):
        print(f"\n  Vehicle {i+1}:")
        print(f"    Confidence: {vehicle['score']:.3f}")
        print(f"    BBox: {vehicle['bbox']}")
        
        if i < len(plate_results_per_vehicle):
            plates = plate_results_per_vehicle[i]
            if plates:
                print(f"    License Plates ({len(plates)}):")
                for j, plate in enumerate(plates):
                    print(f"      Plate {j+1}: '{plate['text']}' (conf: {plate['confidence']:.3f})")
            else:
                print(f"    No license plates detected")
    print(f"{'='*60}\n")
