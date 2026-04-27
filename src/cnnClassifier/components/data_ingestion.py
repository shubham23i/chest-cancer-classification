import os
import urllib.request as request
import zipfile
import gdown
from cnnClassifier import logger
from cnnClassifier.utils.common import get_size
from cnnClassifier.entity.config_entity import DataIngestionConfig  


class DataIngestion:
    def __init__(self, config):
        self.config = config

    def download_file(self):
        try:
            dataset_url = self.config.source_URL
            download_dir = self.config.unzip_dir

            os.makedirs(download_dir, exist_ok=True)

            logger.info(f"Downloading data from {dataset_url} into {download_dir}")

            gdown.download_folder(
                url=dataset_url,
                output=download_dir,
                quiet=False,
                resume=True
            )

            logger.info(f"Downloaded data into {download_dir}")

        except Exception as e:
            logger.exception(e)
            raise e