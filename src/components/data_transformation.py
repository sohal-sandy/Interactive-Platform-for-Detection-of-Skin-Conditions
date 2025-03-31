import os
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision import datasets, transforms
from PIL import Image
from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging
from src.utils import save_transform_config
import torch.optim as optim

@dataclass
class DataTransformationConfig:
    processed_data_dir: str = os.path.join("artifacts", "processed_data") #look into why this is necessary
    transform_config_path: str = os.path.join("artifacts", "transform_config.pkl")
    batch_size: int = 64 #try 32
    image_size: tuple = (224, 224)

class DataTransformation:
    def __init__(self):
        self.config = DataTransformationConfig()

        # Define transformations as parameters (not full objects)
        self.train_transform_params = {
            'resize': (224, 224),
            'flip_prob': 0.5,
            'rotation_deg': 20,
            'color_jitter': {'brightness': 0.2, 'contrast': 0.2, 'saturation': 0.2, 'hue': 0.1}
        }

        
        self.test_transform_params = {
            'resize': (224, 224)
        }
    def initiate_data_transformation(self, train_dir, test_dir):
        try:
            logging.info("Loading and transforming image data...")

            # Define training transformations using the parameters
            train_transform = transforms.Compose([
                transforms.RandomResizedCrop(self.train_transform_params['resize'], scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(p=self.train_transform_params['flip_prob']),
                transforms.RandomRotation(self.train_transform_params['rotation_deg']),
                transforms.ColorJitter(**self.train_transform_params['color_jitter']),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

            # Define test transformations
            test_transform = transforms.Compose([
                transforms.Resize(self.test_transform_params['resize']),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

            # Load datasets with respective transformations
            train_dataset = datasets.ImageFolder(root=train_dir, transform=train_transform)
            test_dataset = datasets.ImageFolder(root=test_dir, transform=test_transform)

            # Create DataLoaders
            train_loader = DataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=self.config.batch_size, shuffle=False)

            logging.info(f"Image data transformation completed successfully.")

            # Save transformation configuration (only the parameters)
            save_transform_config({
                "train_transform_params": self.train_transform_params,
                "test_transform_params": self.test_transform_params
            }, self.config.transform_config_path)

            return train_loader, test_loader, self.config.transform_config_path

        except Exception as e:
            raise CustomException(e, sys)