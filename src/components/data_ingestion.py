import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
import shutil


from sklearn.model_selection import train_test_split
from dataclasses import dataclass

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig

from src.components.model_trainer import ModelTrainerConfig
from src.components.model_trainer import ModelTrainer
from src.utils import evaluate_model

@dataclass #decorator instead of __init__
class DataIngestionConfig:
    artifacts_dir: str = os.path.join("artifacts")
    raw_data_dir: str = os.path.join("notebook", "data", "archive")  # Path to original dataset
    processed_data_dir: str = os.path.join(artifacts_dir, "processed_data")  # Train/Test split storage

    train_dir: str = os.path.join(processed_data_dir, "train")
    test_dir: str = os.path.join(processed_data_dir, "test")

    #for csv can be 
    #train_data_path: str=os.path.join('artifacts', 'train.csv')
    #test_data_path: str=os.path.join('artifacts', 'test.csv')
    #raw_data_path: str=os.path.join('artifacts', 'data.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig() #the variable above will be strored in this variable

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try: #read from source...if you want to source from mongodb modify the code below.
            # Ensure artifacts/raw_data directory exists
            raw_data_target_dir = os.path.join(self.ingestion_config.artifacts_dir, "raw_data")
            os.makedirs(raw_data_target_dir, exist_ok=True)

            # Ensure processed data directory exists
            os.makedirs(self.ingestion_config.processed_data_dir, exist_ok=True)

            # Define original dataset paths
            original_train_dir = os.path.join(self.ingestion_config.raw_data_dir, "train")
            original_test_dir = os.path.join(self.ingestion_config.raw_data_dir, "test")

            # Define new target locations inside artifacts/raw_data/
            target_train_dir = os.path.join(raw_data_target_dir, "train")
            target_test_dir = os.path.join(raw_data_target_dir, "test")


            # Copy train images to artifacts/raw_data/train
            if os.path.exists(original_train_dir):
                shutil.copytree(original_train_dir, target_train_dir, dirs_exist_ok=True)
                logging.info(f"Copied training data to {target_train_dir}")

            # Copy test images to artifacts/raw_data/test
            if os.path.exists(original_test_dir):
                shutil.copytree(original_test_dir, target_test_dir, dirs_exist_ok=True)
                logging.info(f"Copied test data to {target_test_dir}")

            # Move data to processed directory
            shutil.copytree(target_train_dir, self.ingestion_config.train_dir, dirs_exist_ok=True)
            shutil.copytree(target_test_dir, self.ingestion_config.test_dir, dirs_exist_ok=True)

            logging.info("Data ingestion for image dataset completed successfully")

            return(
                 self.ingestion_config.train_dir, 
                 self.ingestion_config.test_dir
            )

        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":
    
    try:
        # Step 1: Data Ingestion
        data_ingestion = DataIngestion()
        train_data_dir, test_data_dir = data_ingestion.initiate_data_ingestion()

        # Step 2: Data Transformation
        data_transformation = DataTransformation()
        # You will need to implement image data transformation logic here
        train_loader, test_loader, _ = data_transformation.initiate_data_transformation(train_data_dir, test_data_dir)

        # Step 3: Model Training
        model_trainer = ModelTrainer()
        train_ = model_trainer.train(train_loader, test_loader)

        # Step 4: Model Evaluation
        result = evaluate_model(model_trainer.model, test_loader, model_trainer.config.device)
        
        print(result)

    except Exception as e:
        logging.error(f"Error during the process: {str(e)}")
        raise CustomException(e, sys)


