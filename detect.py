#!/usr/bin/env python3
"""
Fully Standalone Vehicle and License Plate Detection

This script does NOT require the ppdet library at all.
It uses:
- PaddleInference API to load the exported YOLOv3 model
- PaddleOCR for license plate detection and recognition

Usage:
    python detect.py --image path/to/image.jpg --output output/
"""

import os
import argparse
import cv2
import numpy as np
from paddleocr import PaddleOCR
import yaml


def load_vehicle_detector(model_dir="exported_model/vehicle_plate_config"):
    """Load YOLOv3 vehicle detection model using PaddleInference."""
    try:
        from paddle.inference import Config, create_predictor
        
        # Check if model files exist
        model_file = os.path.join(model_dir, "model.json")
        params_file = os.path.join(model_dir, "model.pdiparams")
        
        if not os.path.exists(model_file):
            raise FileNotFoundError(f"Model file not found: {model_file}")
        if not os.path.exists(params_file):
            raise FileNotFoundError(f"Params file not found: {params_file}")
        
        # Create config
        config = Config(model_file, params_file)
        config.disable_gpu()
        config.switch_use_feed_fetch_ops(False)
        
        # Create predictor
        predictor = create_predictor(config)
        
        # Load inference config for preprocessing info
        infer_cfg_path = os.path.join(model_dir, "infer_cfg.yml")
        with open(infer_cfg_path, 'r') as f:
            infer_cfg = yaml.safe_load(f)
        
        return predictor, infer_cfg
        
    except Exception as e:
        print(f"Error loading model: {e}")
        raise


def preprocess_image(image, target_size=608):
    """Preprocess image for YOLOv3 inference."""
    # Get original shape
    orig_h, orig_w = image.shape[:2]
    
    # Resize to target size (608x608) without keeping aspect ratio
    resized = cv2.resize(image, (target_size, target_size))
    
    # Convert to RGB and normalize with ImageNet mean/std
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32)
    mean = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 3)
    std = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 3)
    normalized = (rgb / 255.0 - mean) / std
    
    # Transpose to CHW format
    transposed = normalized.transpose(2, 0, 1)
    
    # Add batch dimension
    batched = np.expand_dims(transposed, axis=0).astype(np.float32)
    
    # Prepare im_shape and scale_factor
    im_shape = np.array([[orig_h, orig_w]], dtype=np.float32)
    scale_factor = np.array([[target_size / orig_h, target_size / orig_w]], dtype=np.float32)
    
    return batched, im_shape, scale_factor


def postprocess_detections(bbox_data, bbox_num, im_shape, scale_factor, score_threshold=0.2):
    """Post-process YOLOv3 outputs to get bounding boxes."""
    vehicles = []
    
    try:
        if bbox_data is None or len(bbox_data) == 0:
            return vehicles
        
        # Extract scale factors
        scale_h, scale_w = scale_factor[0]
        
        # bbox_data format: [class_id, score, x1, y1, x2, y2]
        # The model outputs coordinates in an upscaled space
        # Multiply by scale_factor to get original image coordinates
        for detection in bbox_data:
            if len(detection) >= 6:
                class_id, score, x1, y1, x2, y2 = detection[:6]
                
                if score >= score_threshold:
                    # Scale coordinates to original image space
                    # Model outputs in upscaled space, multiply by scale to get original
                    x1_orig = x1 * scale_w
                    y1_orig = y1 * scale_h
                    x2_orig = x2 * scale_w
                    y2_orig = y2 * scale_h
                    
                    vehicles.append({
                        'bbox': [float(x1_orig), float(y1_orig), float(x2_orig), float(y2_orig)],
                        'score': float(score),
                        'class_id': int(class_id)
                    })
    
    except Exception as e:
        print(f"Error in postprocessing: {e}")
        import traceback
        traceback.print_exc()
    
    return vehicles


def detect_vehicles(predictor, image, score_threshold=0.2):
    """Detect vehicles in image using PaddleInference."""
    # Preprocess
    input_image, im_shape, scale_factor = preprocess_image(image)
    
    # Get input handles
    input_names = predictor.get_input_names()
    
    # Set inputs (order: im_shape, image, scale_factor based on get_input_names output)
    for name in input_names:
        handle = predictor.get_input_handle(name)
        if name == 'image':
            handle.copy_from_cpu(input_image)
        elif name == 'im_shape':
            handle.copy_from_cpu(im_shape)
        elif name == 'scale_factor':
            handle.copy_from_cpu(scale_factor)
    
    # Run inference
    predictor.run()
    
    # Get outputs
    output_names = predictor.get_output_names()
    bbox_data = predictor.get_output_handle(output_names[0]).copy_to_cpu()
    bbox_num = predictor.get_output_handle(output_names[1]).copy_to_cpu() if len(output_names) > 1 else None
    
    # Post-process with scale information
    vehicles = postprocess_detections(bbox_data, bbox_num, im_shape, scale_factor, score_threshold)
    
    return vehicles


def crop_vehicle_region(image, bbox):
    """Crop vehicle region from image."""
    x1, y1, x2, y2 = map(int, bbox)
    h, w = image.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    return image[y1:y2, x1:x2]


def detect_plates(ocr_engine, vehicle_crop, threshold=0.3):
    """Detect and recognize license plates in vehicle crop."""
    if vehicle_crop is None or vehicle_crop.size == 0:
        return []
    
    try:
        results = ocr_engine.predict(vehicle_crop)
        if not results:
            return []
        
        plates = []
        for result in results:
            if result is None:
                continue
            
            # Parse result
            if isinstance(result, dict):
                bbox = result.get('bbox', result.get('box', None))
                text = result.get('text', result.get('rec_text', ''))
                confidence = result.get('confidence', result.get('rec_score', 0.0))
            elif isinstance(result, (list, tuple)) and len(result) >= 2:
                bbox = result[0]
                if isinstance(result[1], (list, tuple)) and len(result[1]) >= 2:
                    text, confidence = result[1][0], result[1][1]
                else:
                    text = str(result[1])
                    confidence = 0.0
            else:
                continue
            
            if confidence >= threshold:
                plates.append({
                    'bbox': bbox,
                    'text': text,
                    'confidence': confidence
                })
        
        return plates
    except Exception as e:
        print(f"Plate detection error: {e}")
        return []


def visualize_results(image, vehicles, plates_per_vehicle):
    """Draw detection results on image."""
    vis_image = image.copy()
    
    for i, vehicle in enumerate(vehicles):
        # Draw vehicle box (green)
        x1, y1, x2, y2 = map(int, vehicle['bbox'])
        cv2.rectangle(vis_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Draw vehicle label
        label = f"Vehicle: {vehicle['score']:.2f}"
        cv2.putText(vis_image, label, (x1, y1 - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Draw plates
        if i < len(plates_per_vehicle):
            for plate in plates_per_vehicle[i]:
                # Convert to absolute coordinates
                vx1, vy1 = vehicle['bbox'][0], vehicle['bbox'][1]
                abs_bbox = [[p[0] + vx1, p[1] + vy1] for p in plate['bbox']]
                
                # Draw plate box (blue)
                pts = np.array(abs_bbox, dtype=np.int32)
                cv2.polylines(vis_image, [pts], True, (255, 0, 0), 2)
                
                # Draw text
                text_label = f"{plate['text']} ({plate['confidence']:.2f})"
                text_x, text_y = int(abs_bbox[0][0]), int(abs_bbox[0][1]) - 10
                
                # Background for text
                (text_w, text_h), _ = cv2.getTextSize(
                    text_label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(vis_image,
                            (text_x, text_y - text_h - 5),
                            (text_x + text_w, text_y + 5),
                            (0, 0, 255), -1)
                cv2.putText(vis_image, text_label, (text_x, text_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    return vis_image


def main():
    parser = argparse.ArgumentParser(description='Standalone Vehicle and Plate Detection')
    parser.add_argument('--image', required=True, help='Input image path')
    parser.add_argument('--output', default='output', help='Output directory')
    parser.add_argument('--model_dir', default='exported_model/vehicle_plate_config',
                       help='Path to exported model directory')
    parser.add_argument('--vehicle_threshold', type=float, default=0.2,
                       help='Vehicle detection confidence threshold')
    parser.add_argument('--plate_threshold', type=float, default=0.3,
                       help='Plate detection confidence threshold')
    parser.add_argument('--lang', default='ch', help='OCR language (ch/en)')
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Load image
    print(f"Loading image: {args.image}")
    image = cv2.imread(args.image)
    if image is None:
        print(f"Error: Could not read image {args.image}")
        return
    
    # Load vehicle detector
    print(f"Loading vehicle detection model from {args.model_dir}...")
    predictor, infer_cfg = load_vehicle_detector(args.model_dir)
    print("✓ Vehicle detector loaded")
    
    # Initialize OCR
    print("Initializing PaddleOCR...")
    ocr = PaddleOCR(use_textline_orientation=False, lang=args.lang)
    print("✓ PaddleOCR initialized")
    
    # Detect vehicles
    print("\nDetecting vehicles...")
    vehicles = detect_vehicles(predictor, image, args.vehicle_threshold)
    print(f"✓ Found {len(vehicles)} vehicle(s)")
    
    # Detect plates in each vehicle
    plates_per_vehicle = []
    for i, vehicle in enumerate(vehicles):
        print(f"  Processing vehicle {i+1}/{len(vehicles)}...")
        vehicle_crop = crop_vehicle_region(image, vehicle['bbox'])
        plates = detect_plates(ocr, vehicle_crop, args.plate_threshold)
        plates_per_vehicle.append(plates)
        
        if plates:
            print(f"    ✓ Found {len(plates)} plate(s):")
            for plate in plates:
                print(f"      - '{plate['text']}' (confidence: {plate['confidence']:.3f})")
        else:
            print(f"    - No plates detected")
    
    # Visualize
    print("\nCreating visualization...")
    vis_image = visualize_results(image, vehicles, plates_per_vehicle)
    
    # Save
    output_path = os.path.join(args.output, os.path.basename(args.image))
    cv2.imwrite(output_path, vis_image)
    print(f"✓ Saved result to: {output_path}")
    
    print("\n✅ Done!")


if __name__ == '__main__':
    main()
