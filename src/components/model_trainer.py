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


#evaluate added
#evaluate class moved to utils.py
@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join("artifacts", "resnet18.pth")
    num_epochs: int = 15
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


            model_report = evaluate_model(model=self.model, test_loader=test_loader, device=self.config.device)




            logging.info("Training complete. Saving model...")
            torch.save(self.model.state_dict(), self.config.trained_model_file_path)
#from save changed to save_object
        except Exception as e:
            raise CustomException(e, sys)

    