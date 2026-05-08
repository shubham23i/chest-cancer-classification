import os
import zipfile
import gdown
import shutil
from cnnClassifier import logger


class DataIngestion:
    def __init__(self, config):
        self.config = config

    def download_file(self):
        try:
            file_id = "1UhQi-Tm_nESX-AgNEmTEWw_XKFhAWc1m"
            zip_path = self.config.local_data_file

            os.makedirs(self.config.root_dir, exist_ok=True)

            logger.info("Downloading ZIP file...")
            gdown.download(
                f"https://drive.google.com/uc?/export=download&id={file_id}",
                zip_path,
                quiet=False
            )

            logger.info(f"Downloaded at {zip_path}")

        except Exception as e:
            logger.exception(e)
            raise e

    def extract_zip_file(self):
        try:
            unzip_path = self.config.unzip_dir
            zip_path = self.config.local_data_file

            logger.info("Extracting ZIP file...")

            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(unzip_path)

            logger.info("Extraction completed")

        except Exception as e:
            logger.exception(e)
            raise e

    def create_binary_dataset(self):
        root_dir = self.config.unzip_dir
        data_dir = os.path.join(root_dir, "Data")

        carcinoma_classes = [
            "adenocarcinoma_left.lower.lobe_T2_N0_M0_Ib",
            "large.cell.carcinoma_left.hilum_T2_N2_M0_IIIa",
            "squamous.cell.carcinoma_left.hilum_T1_N2_M0_IIIa"
        ]

        splits = ["train", "test", "valid"]

        for split in splits:
            split_path = os.path.join(data_dir, split)

            carcinoma_target = os.path.join(split_path, "carcinoma")
            os.makedirs(carcinoma_target, exist_ok=True)

            for cls in carcinoma_classes:
                src_folder = os.path.join(split_path, cls)

                if not os.path.exists(src_folder):
                    continue

                for file in os.listdir(src_folder):
                    shutil.move(
                        os.path.join(src_folder, file),
                        os.path.join(carcinoma_target, file)
                    )

                os.rmdir(src_folder)

        logger.info("Binary dataset created successfully")