import fire
from loguru import logger

from omegaconf import OmegaConf
from src.data_prep.ingest_data import main as ingest_data_fun
from src.data_prep.create_composite import create_and_export_comp_dataset
import os


class Pipeline(object):

    def ingest_data(self, config_path):
        config = OmegaConf.load(config_path)
        logger.info(f"Config : {config}")
        for data_subset in os.listdir(config.DATA.RAW):
            self.ingest_data_subset(config_path, data_subset)

    def ingest_data_subset(self, config_path, data_subset):
        config = OmegaConf.load(config_path)
        logger.info(f"Config : {config}")
        dataset_local_path = os.path.join(config.DATA.RAW, data_subset)
        embedding_center_path = config.LOCAL_EMBEDDING_CENTER.local_path
        ingested_data_path = os.path.join(config.DATA.INGESTED, data_subset)
        logger.info(f"Ingest data from {dataset_local_path}")
        ingest_data_fun(dataset_local_path,
                        embedding_center_path,
                        ingested_data_path)
        logger.info(f"Save ingested data to {ingested_data_path}")

    def create_composite(self, config_path):
        config = OmegaConf.load(config_path)
        logger.info(f"Config : {config}")
        for data_subset in os.listdir(config.DATA.INGESTED):
            self.create_composite_subset(config_path, data_subset)

    def create_composite_subset(self, config_path, data_subset):
        config = OmegaConf.load(config_path)
        logger.info(f"Config : {config}")
        ingested_data_path = os.path.join(str(config.DATA.INGESTED), data_subset)
        composite_data_path = os.path.join(str(config.DATA.COMPOSITE), data_subset)
        logger.info(f"create_and_export_comp_dataset from {ingested_data_path}")
        create_and_export_comp_dataset(ingested_data_path, composite_data_path)


if __name__ == '__main__':
    fire.Fire(Pipeline)

