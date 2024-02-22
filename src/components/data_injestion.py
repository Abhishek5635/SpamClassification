import os
import sys
from src.logger import logging
from src.exception import CustomException
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

## Initialize the Data Injestion Configuration

@dataclass
class DataInjestionconfig:
    train_data_path = os.path.join("artifacts","train.csv")
    test_data_path = os.path.join("artifacts","test.csv")


## create a class for Data Injestion
class DataInjestion:
    def __init__(self):
        self.ingestion_config = DataInjestionconfig()

    
    def initiate_data_injestion(self):
        logging.info("Starting Data Injestion")

        try:
            df = pd.read_csv(os.path.join("D:\Spam Ham Classification\\notebook\data\\","model_build_data.csv"))
            logging.info("Train Test Split")
            train_set, test_set = train_test_split(df,test_size=0.20,random_state=2)
            train_set.to_csv(self.ingestion_config.train_data_path,index=False)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False)


            logging.info("Ingestion of data is completed")

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

            
        except Exception as e:
            logging.info("Exception occoured at Data Injestion stage")
            raise CustomException(e,sys)
        

if __name__ == "__main__":
    obj = DataInjestion()
    train_data_path, test_data_path = obj.initiate_data_injestion()