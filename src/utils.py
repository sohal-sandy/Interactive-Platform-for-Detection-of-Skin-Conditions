import os
import sys
import torch
import numpy as np 
import pandas as pd
import dill
import pickle
import logging
from src.exception import CustomException
from torch.utils.data import DataLoader
import torchvision.models as models

def save_object(obj, file_path):
    try:
        torch.save(obj.state_dict(), file_path)  # Save the state_dict of the model
        print(f"Model saved to {file_path}")
    except Exception as e:
        raise CustomException(f"Error saving object: {e}",sys)


def save_transform_config(transform_config, file_path):
    try:
        with open(file_path, 'wb') as file:
            pickle.dump(transform_config, file)
        print(f"Transformation config saved to {file_path}")
    except Exception as e:
        raise CustomException(f"Error saving transformation config: {e}", sys)


def load_transform_config(file_path):
    try:
        with open(file_path, 'rb') as file:
            return pickle.load(file)
    except Exception as e:
        raise CustomException(f"Error loading transformation config: {e}",sys)


def evaluate_model(model, test_loader: DataLoader, device: str) -> dict:
    """Evaluate the trained model on test data and return a report."""
    try:
        logging.info("Evaluating model...")
        model.eval()  # Set model to evaluation mode
        correct, total = 0, 0
            
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

        accuracy = 100 * correct / total
        logging.info(f"Test Accuracy: {accuracy:.2f}%")
        return {"accuracy" : accuracy}

    except Exception as e:
        raise CustomException(e, sys)


def load_object(file_path, is_model=False, architecture=None, num_classes=4):
    try:
        if is_model:
            # Check if model architecture is provided
            if architecture is None:
                raise ValueError("Model architecture must be provided if is_model=True")
            
            # Create the model architecture instance
            model = architecture()

            # Modify the final fully connected layer to match the number of classes
            in_features = model.fc.in_features  # Get the input features to the final layer
            model.fc = torch.nn.Linear(in_features, num_classes)  # Adjust the number of classes

            # Load the model weights
            model.load_state_dict(torch.load(file_path, weights_only=True))
            model.eval()  # Set to evaluation mode
            return model
        
        else:
            # Handle non-model object loading (e.g., configs)
            with open(file_path, 'rb') as file:
                return torch.load(file)
    
    except Exception as e:
        raise CustomException(f"Error loading object from {file_path}: {str(e)}", sys)