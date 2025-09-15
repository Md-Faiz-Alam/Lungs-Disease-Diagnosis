from datetime import datetime
from typing import List

import torch

# General
TIMESTAMP: str = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")

# Data Ingestion Constants
ARTIFACT_DIR: str = "artifacts"
BUCKET_NAME: str = "lungxray"
S3_DATA_FOLDER: str = "data"

# Class Labels
CLASS_LABEL_1: str = "NORMAL"
CLASS_LABEL_2: str = "PNEUMONIA"

# Image Transform Parameters
BRIGHTNESS: float = 0.10
CONTRAST: float = 0.1
SATURATION: float = 0.10
HUE: float = 0.1
RESIZE: int = 224
CENTERCROP: int = 224
RANDOMROTATION: int = 10
NORMALIZE_LIST_1: List[float] = [0.485, 0.456, 0.406]
NORMALIZE_LIST_2: List[float] = [0.229, 0.224, 0.225]

# Transform Files
TRAIN_TRANSFORMS_KEY: str = "xray_train_transforms"
TRAIN_TRANSFORMS_FILE: str = "train_transforms.pkl"
TEST_TRANSFORMS_FILE: str = "test_transforms.pkl"

# DataLoader Params
BATCH_SIZE: int = 2
SHUFFLE: bool = False
PIN_MEMORY: bool = True

# Model Training Constants
TRAINED_MODEL_DIR: str = "trained_model"
TRAINED_MODEL_NAME: str = "model.pt"
DEVICE: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
STEP_SIZE: int = 6
GAMMA: float = 0.5
EPOCH: int = 10

# BentoML
BENTOML_MODEL_NAME: str = "xray_model"
BENTOML_SERVICE_NAME: str = "xray_service"
BENTOML_ECR_IMAGE: str = "xray_bento_image"

# Prediction Labels
PREDICTION_LABEL: dict = {0: CLASS_LABEL_1, 1: CLASS_LABEL_2}
