import os
import sys
import torch
from PIL import Image
from torchvision import transforms
from src.exception import CustomException
from src.utils import load_object
import torchvision.models as models
from src.utils import load_transform_config

class PredictPipeline:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def predict(self, image_path: str):
        try:
            model_path = os.path.join("artifacts", "resnet18.pth")
            preprocessor_path = os.path.join('artifacts', 'transform_config.pkl')
            print("Before Loading")
            
            # Load model and transformation parameters
            model = load_object(file_path=model_path, is_model=True, architecture=models.resnet18, num_classes=4)  # Set num_classes to 4
            model.to(self.device)
            model.eval() #set model to evaluation mode

            transform_params = load_transform_config(file_path=preprocessor_path)
            
            print("After Loading")
            
            # Recreate transformations from parameters
            test_transform = transforms.Compose([
                transforms.Resize(transform_params['test_transform_params']['resize']),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            # Preprocess the image
            image = Image.open(image_path).convert("RGB")  # Open image as a PIL Image
            
            # Apply the test transformation to the PIL image
            image_tensor_scaled = test_transform(image).unsqueeze(0)  # Add batch dimension

            # Make prediction
            with torch.no_grad():
                outputs = model(image_tensor_scaled.to(self.device))  # Pass to model
                probabilities = torch.softmax(outputs, dim=1)
                confidence_score, predicted_class = torch.max(probabilities, 1)

            class_names = ["Acne_Rosacea", "Bullous Disease", "Eczema" , "Warts Molluscum_Viral Infections"]
            
            return class_names[predicted_class.item()], round(confidence_score.item() * 100, 2)
        
        except Exception as e:
            raise CustomException(f"Error during prediction: {str(e)}", sys)
