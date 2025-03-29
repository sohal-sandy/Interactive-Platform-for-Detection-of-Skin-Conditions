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
from src.utils import save_object
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
        
        # Define training transformations
        self.train_transform = transforms.Compose([
                transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),  # Randomly crop and resize
                transforms.RandomHorizontalFlip(p=0.5),  # Random horizontal flip
                transforms.RandomRotation(20),  # Random rotation (-20° to 20°)
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Color variations
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Standard normalization for RGB Channels
        ])
        
        # Define test transformations
        self.test_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def initiate_data_transformation(self, train_dir, test_dir):
        try:
            train_dir = r"C:\Users\sande\Desktop\Git\skin_app\notebook\data\archive\train"
            test_dir = r"C:\Users\sande\Desktop\Git\skin_app\notebook\data\archive\test"

            logging.info("Loading and transforming image data...")
            
            # Load datasets with respective transformations
            train_dataset = datasets.ImageFolder(root=train_dir, transform=self.train_transform)
            test_dataset = datasets.ImageFolder(root=test_dir, transform=self.test_transform)
            
            # Create DataLoaders
            train_loader = DataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=self.config.batch_size, shuffle=False)
            
            logging.info(f"Image data transformation completed successfully.")
            
            # Save transformation configuration
            save_object(self.config.transform_config_path, {
                "train_transform": self.train_transform,
                "test_transform": self.test_transform
            })
            
            return train_loader, test_loader, self.config.transform_config_path
        
        except Exception as e:
            raise CustomException(e, sys)
