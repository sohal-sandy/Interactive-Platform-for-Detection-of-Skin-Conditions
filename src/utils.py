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

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    


def evaluate_model(model, test_loader: DataLoader, device: str) -> dict: ##
    """Evaluate the trained model on test data and return a report."""
    try:
        logging.info("Evaluating ResNet18 model...")
        model.eval()
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
        return {"accuracy" : accuracy}  ##

    except Exception as e:
        raise CustomException(e, sys)
