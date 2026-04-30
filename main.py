from cnnClassifier import logger
from cnnClassifier.pipeline.stage_01_data_ingestion import data_ingestion_pipeline
from cnnClassifier.pipeline.stage_02_prepare_base_model import PrepareBaseModelTrainingPipeline
from cnnClassifier.pipeline.stage_03_model_trainer import ModelTrainingPipeline

stage_name="data ingestion stage"
try:
    logger.info(f"Starting {stage_name}")
    obj=data_ingestion_pipeline()
    obj.main()
    logger.info(f"Finished {stage_name}")
except Exception as e:
    logger.exception(e)
    raise e 


STAGE_NAME="prepare base model"
try:
    logger.info("*******************")
    logger.info(f">>>>> stage {STAGE_NAME} started <<<<<<")
    obj = PrepareBaseModelTrainingPipeline()
    obj.main()
    logger.info(f">>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e

STAGE_NAME="MODEL_TRAINER"

try:
    logger.info(f"****************")
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    obj = ModelTrainingPipeline()
    obj.main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e