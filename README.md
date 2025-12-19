# Vehicle & License Plate Detection - Minimal Standalone

A minimal, standalone vehicle detection and license plate recognition system using PaddleInference and PaddleOCR.

## Features

- **Fully Standalone**: No ppdet library required
- **Vehicle Detection**: YOLOv3 model via PaddleInference API
- **License Plate Recognition**: PP-OCRv5 for detection and text recognition
- **Minimal Dependencies**: Only essential packages

## Quick Start

### Installation

```bash
pip install -r requirements-minimal.txt
```

### Usage

**Detect vehicles and recognize license plates:**

```bash
python detect.py --image path/to/image.jpg --output output/
```

### Parameters

- `--image` - Input image path (required)
- `--output` - Output directory (default: `output`)
- `--model_dir` - Path to exported model (default: `exported_model/vehicle_plate_config`)
- `--vehicle_threshold` - Vehicle detection confidence (default: 0.2)
- `--plate_threshold` - Plate detection confidence (default: 0.3)
- `--lang` - OCR language: 'ch' or 'en' (default: 'ch')

## How It Works

1. **Load Model**: YOLOv3 exported model loaded via PaddleInference API
2. **Detect Vehicles**: Find all vehicles in the image
3. **Crop Vehicles**: Extract each vehicle region
4. **Detect Plates**: Run PaddleOCR on each vehicle crop
5. **Recognize Text**: Extract plate text and confidence
6. **Visualize**: Draw boxes and text on image
7. **Save**: Output annotated image

## Project Structure

```
ppvehicle/
├── detect.py                    # Main standalone detection script
├── exported_model/              # Exported YOLOv3 model
│   └── vehicle_plate_config/
│       ├── model.json           # Model structure
│       ├── model.pdiparams      # Model weights (235MB)
│       └── infer_cfg.yml        # Inference config
├── demo/                        # Demo images
├── requirements-minimal.txt     # Minimal dependencies
└── README.md                    # This file
```

## Dependencies

- Python 3.7+
- PaddlePaddle
- PaddleOCR
- OpenCV
- NumPy
- PyYAML

See `requirements-minimal.txt` for exact versions.

## Exporting Custom Models (Advanced)

If you have a custom PaddleDetection model or want to export a different model, follow these steps:

### Prerequisites

You'll need the full PaddleDetection library temporarily for model export:

```bash
# Clone PaddleDetection (or use your existing setup)
git clone https://github.com/PaddlePaddle/PaddleDetection.git
cd PaddleDetection
pip install -r requirements.txt
```

### Export Script

Create an export script (e.g., `export_model.py`):

```python
import sys
import os
from ppdet.core.workspace import load_config, merge_config
from ppdet.engine import Trainer
import paddle

def export_model(config_path, weights_path, output_dir):
    """Export PaddleDetection model to inference format."""
    # Load config
    cfg = load_config(config_path)
    cfg.use_gpu = False
    
    # Set device
    paddle.set_device('cpu')
    
    # Create trainer
    trainer = Trainer(cfg, mode='test')
    
    # Load weights
    print(f"Loading weights from {weights_path}...")
    trainer.load_weights(weights_path)
    
    # Export model
    print(f"Exporting model to {output_dir}...")
    trainer.export(output_dir=output_dir)
    
    print(f"\n✅ Model exported successfully!")
    print(f"   Location: {output_dir}/")
    print(f"   Files: model.json, model.pdiparams, infer_cfg.yml")

if __name__ == '__main__':
    # Example: Export YOLOv3 vehicle model
    export_model(
        config_path='configs/ppvehicle/vehicle_yolov3/vehicle_yolov3_darknet.yml',
        weights_path='https://paddledet.bj.bcebos.com/models/vehicle_yolov3_darknet.pdparams',
        output_dir='exported_model'
    )
```

### Run Export

```bash
python export_model.py
```

### Output Files

The export creates three files:
- `model.json` - Model structure (for PaddleInference)
- `model.pdiparams` - Model weights
- `infer_cfg.yml` - Inference configuration (preprocessing info)

### Using the Exported Model

Update the `--model_dir` parameter in `detect.py`:

```bash
python detect.py \
  --image demo/car.jpg \
  --model_dir path/to/your/exported_model \
  --output output/
```

### Important Notes

1. **Model Format**: The export creates PaddleInference format (`.json` + `.pdiparams`), not the older `.pdmodel` format
2. **Preprocessing**: The `infer_cfg.yml` contains preprocessing parameters - ensure your `detect.py` matches these
3. **Input Names**: Different models may have different input names - check with:
   ```python
   from paddle.inference import Config, create_predictor
   config = Config('model.json', 'model.pdiparams')
   predictor = create_predictor(config)
   print(predictor.get_input_names())
   ```
4. **Cleanup**: After exporting, you can remove the PaddleDetection library to keep your deployment minimal



## Model Information

- **Vehicle Detection**: YOLOv3 DarkNet trained on PPVehicle dataset
  - 6 classes: car, truck, bus, motorbike, tricycle, carplate
  - Input size: 608x608
  - Exported to PaddleInference format
- **Plate Detection**: PP-OCRv5 server detection model (auto-downloaded)
- **Text Recognition**: PP-OCRv5 server recognition model (auto-downloaded)

## Output Format

- **Images**: Green boxes for vehicles, blue boxes for plates
- **Console**: Detection results with confidence scores
- **Plate Text**: Displayed on image with confidence

## Example

```bash
python detect.py --image demo/car.jpg --output output/
```

Output:
```
Loading image: demo/car.jpg
Loading vehicle detection model...
✓ Vehicle detector loaded
✓ PaddleOCR initialized

Detecting vehicles...
✓ Found 3 vehicle(s)
  Processing vehicle 1/3...
    - No plates detected
  Processing vehicle 2/3...
    - No plates detected
  Processing vehicle 3/3...
    - No plates detected

✓ Saved result to: output/car.jpg
✅ Done!
```

## License

Apache License 2.0

## Acknowledgments

- [PaddlePaddle](https://github.com/PaddlePaddle/Paddle) - Deep learning framework
- [PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection) - Original detection framework
- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) - OCR toolkit
- PPVehicle dataset (BDD100K-MOT + UA-DETRAC)
