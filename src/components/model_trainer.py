import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataclasses import dataclass
import torchvision.models as models
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_model
from src.components.data_transformation import DataTransformation


@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join("artifacts", "resnet18.pth")
    num_epochs: int = 1
    learning_rate: float = 0.001
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

class ModelTrainer:
    def __init__(self, num_classes=4):
        self.config = ModelTrainerConfig()

        # Load pre-trained ResNet18 and modify the last layer
        self.model = models.resnet18(pretrained=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        self.model = self.model.to(self.config.device)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.learning_rate)

    def train(self, train_loader: DataLoader, test_loader: DataLoader):
        """Train the model using the provided DataLoader"""
        try:
            logging.info("Starting ResNet18 model training...")
            self.model.train()
            for epoch in range(self.config.num_epochs):
                total_loss = 0.0
                correct, total = 0, 0

                for images, labels in train_loader:
                    images, labels = images.to(self.config.device), labels.to(self.config.device)
                    
                    self.optimizer.zero_grad()
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                    loss.backward()
                    self.optimizer.step()
                    
                    total_loss += loss.item()
                    _, predicted = torch.max(outputs, 1)
                    correct += (predicted == labels).sum().item()
                    total += labels.size(0)

                epoch_loss = total_loss / len(train_loader)
                epoch_acc = 100 * correct / total
                logging.info(f"Epoch {epoch+1}/{self.config.num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%")

            # Evaluating the model after training
            model_report = evaluate_model(model=self.model, test_loader=test_loader, device=self.config.device)

            # Log evaluation results
            logging.info(f"Model Evaluation: {model_report}")

            # Saving the trained model using the custom save_object function
            logging.info("Training complete. Saving model...")
            save_object(self.model, self.config.trained_model_file_path)

        except Exception as e:
            raise CustomException(e, sys)

    def load_trained_model(self):
        """Load the trained model using the custom load_object function."""
        try:
            # Load the model architecture first
            model = models.resnet18(pretrained=False)  # Set pretrained=False to avoid reloading weights
            model.fc = nn.Linear(model.fc.in_features, 4)  # Adjust for the number of classes (4 in your case)

            # Load the state_dict into the model
            model = load_object(self.config.trained_model_file_path, is_model=True, model=model)
            logging.info(f"Model loaded successfully from {self.config.trained_model_file_path}")
            return model
        except Exception as e:
            raise CustomException(f"Error loading model: {e}", sys)


    