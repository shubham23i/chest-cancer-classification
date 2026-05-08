from cnnClassifier import logger
from cnnClassifier.components.model_evaluation_with_mlflow import Evaluation

from cnnClassifier.utils.common import CONFIG_FILE_PATH, PARAMS_FILE_PATH
from cnnClassifier.config.configuration import ConfigurationManager


stage_name = "Model Evaluation"

class EvaluationPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager(
            config_file_path=CONFIG_FILE_PATH,
            params_file_path=PARAMS_FILE_PATH
        )
        eval_config = config.get_evaluation_config()
        evaluation = Evaluation(eval_config)
        evaluation.evaluation()
        #evaluation.log_into_mlflow()

if __name__ == "__main__":
    try:
        logger.info(f"********************")
        logger.info(f">>>>>> stage {stage_name} started <<<<<<")
        
        obj = EvaluationPipeline()
        obj.main()
        
        logger.info(f">>>>>> stage {stage_name} completed <<<<<<\n\nx==========x")
    
    except Exception as e:
        logger.exception(e)
        raise e
    

