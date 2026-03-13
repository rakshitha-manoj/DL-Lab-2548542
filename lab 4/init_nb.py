import nbformat as nbf
import os

nb = nbf.v4.new_notebook()

cells = []

# Title
cells.append(nbf.v4.new_markdown_cell("# African Wildlife Detection: YOLOv5 & YOLOv8 Ensemble\nThis notebook demonstrates training YOLOv5 and YOLOv8 models on the African Wildlife dataset and combining them using Ensemble learning."))

# Setup
cells.append(nbf.v4.new_markdown_cell("## 1. Setup and Dependencies\nInstalling and importing necessary libraries."))
cells.append(nbf.v4.new_code_cell("import os\nimport yaml\nimport cv2\nimport matplotlib.pyplot as plt\nimport numpy as np\nfrom ultralytics import YOLO\nimport glob\nfrom ensemble_boxes import *"))

# Data Preparation
cells.append(nbf.v4.new_markdown_cell("## 2. Data Preparation\nDownloading and organizing the dataset."))
cells.append(nbf.v4.new_code_cell("# Dataset is already downloaded and structured in 'images/' and 'labels/'\nwith open('african-wildlife.yaml', 'r') as f:\n    data_config = yaml.safe_load(f)\nprint(data_config)"))

# YOLOv5 Training
cells.append(nbf.v4.new_markdown_cell("## 3. YOLOv5 Training\nTraining YOLOv5s for object detection."))
cells.append(nbf.v4.new_code_cell("# Training command (run in terminal or here)\n# !yolo task=detect mode=train model=yolov5s.pt data=african-wildlife.yaml epochs=10 imgsz=416 batch=16 name=yolov5_african_wildlife"))

# YOLOv8 Training
cells.append(nbf.v4.new_markdown_cell("## 4. YOLOv8 Training\nTraining YOLOv8s for object detection."))
cells.append(nbf.v4.new_code_cell("# Training command\n# !yolo task=detect mode=train model=yolov8s.pt data=african-wildlife.yaml epochs=10 imgsz=416 batch=16 name=yolov8_african_wildlife"))

# Evaluation
cells.append(nbf.v4.new_markdown_cell("## 5. Model Evaluation\nEvaluating both models on the test set."))
cells.append(nbf.v4.new_code_cell("# Load and evaluate models\nv5_model = YOLO('runs/detect/yolov5_african_wildlife/weights/best.pt')\nv8_model = YOLO('runs/detect/yolov8_african_wildlife/weights/best.pt')\n\n# v5_results = v5_model.val(data='african-wildlife.yaml', split='test')\n# v8_results = v8_model.val(data='african-wildlife.yaml', split='test')"))

# Ensemble
cells.append(nbf.v4.new_markdown_cell("## 6. Ensemble Learning (WBF)\nCombining predictions from YOLOv5 and YOLOv8 using Weighted Box Fusion."))
cells.append(nbf.v4.new_code_cell("# Logic for WBF will be added here"))

# Results Visualization
cells.append(nbf.v4.new_markdown_cell("## 7. Visualization and Insights\nComparing individual models vs Ensemble."))
cells.append(nbf.v4.new_code_cell("# Display sample predictions"))

nb['cells'] = cells

with open('AfricanWildlifeCNN.ipynb', 'w') as f:
    nbf.write(nb, f)

print("Notebook created successfully.")
