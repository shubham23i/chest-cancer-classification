from cnnClassifier import logger
from cnnClassifier.pipeline.stage_01_data_ingestion import data_ingestion_pipeline


try:
    logger.info(f"Starting {stage_name}")
    obj=data_ingestion_pipeline()
    obj.main()
    logger.info(f"Finished {stage_name}")
except Exception as e:
    logger.exception(e)
    raise e 