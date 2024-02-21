import fire
from loguru import logger

from omegaconf import OmegaConf
from src.data_prep.ingest_data import main as ingest_data_fun
from src.data_prep.create_composite import create_and_export_comp_dataset
import os


class Pipeline(object):
    def ingest_data(self, config_path, data_subset):
        config = OmegaConf.load(config_path)
        logger.info(f"Config : {config}")

        def _run(ds):
            dataset_local_path = os.path.join(config.DATA.RAW, ds)
            embedding_center_path = os.path.join(config.LOCAL_EMBEDDING_CENTER.local_path, ds)
            ingested_data_path = os.path.join(config.DATA.INGESTED, ds)
            logger.info(f"Ingest data from {dataset_local_path}")
            ingest_data_fun(dataset_local_path,
                            embedding_center_path,
                            ingested_data_path)
            logger.info(f"Save ingested data to {ingested_data_path}")

        if data_subset.lower() == "all":
            for data_subset in os.listdir(config.DATA.RAW):
                _run(data_subset)
        else:
            _run(data_subset)

    def create_composite(self, config_path, data_subset):
        config = OmegaConf.load(config_path)
        logger.info(f"Config : {config}")

        def _run(ds):
            ingested_data_path = os.path.join(str(config.DATA.INGESTED), ds)
            composite_data_path = os.path.join(str(config.DATA.COMPOSITE), ds)
            logger.info(f"create_and_export_comp_dataset from {ingested_data_path}")
            create_and_export_comp_dataset(ingested_data_path, composite_data_path)

        if data_subset.lower() == "all":
            for data_subset in os.listdir(config.DATA.INGESTED):
                _run(data_subset)
        else:
            _run(data_subset)


if __name__ == '__main__':
    fire.Fire(Pipeline)
    # pipe("./configs/config_lot11-20-11-2023-eau.yaml")
