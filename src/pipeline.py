import fire
from loguru import logger

from omegaconf import OmegaConf
from src.data_prep.ingest_data import main as ingest_data_fun
from src.data_prep.create_composite import create_and_export_comp_dataset
import os

from src.labkit_labeling.create_tasks import create_tasks_lot5_10, create_tasks_subset, create_tasks_lot4, \
    create_tasks_lot1_3, create_tasks_with_sampling
from src.labkit_labeling.matching_old_names_with_new import main_lot3, main_lot5_10, main_reannot_lot1_4_beni, \
    main_reannot_lot1_4_sed, main_lot4
from src.labkit_labeling.generate_annotated_dataset import main as generate_annotated_dataset


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

    def create_tasks(self, config_path, scenario):
        config = OmegaConf.load(config_path)
        logger.info(f"Config : {config}")
        if scenario == "lot4":
            create_tasks_lot4(config.DATA.COMPOSITE, config.DATA.TO_ANNOTATE)
        if scenario == "lot5_10":
            create_tasks_lot5_10(config.DATA.COMPOSITE, config.DATA.TO_ANNOTATE)
        if scenario == "lot1_3":
            create_tasks_lot1_3(config.DATA.COMPOSITE, config.DATA.TO_ANNOTATE)
        if scenario == "new_lot":
            lots = [config.CREATE_TASKS.lot]
            create_tasks_with_sampling(config.DATA.COMPOSITE,
                                       lots,
                                       config.DATA.TO_ANNOTATE,
                                       config.CREATE_TASKS.sampling_type,
                                       sampling_rate=config.CREATE_TASKS.sampling_rate,
                                       n_clusters=config.CREATE_TASKS.n_clusters,
                                       max_samples=config.CREATE_TASKS.max_samples,
                                       save_plot=config.CREATE_TASKS.save_plot)
        else:
            pass

    def matching_old_names_with_new(self, scenario):
        if scenario == "lot3":
            main_lot3()
        if scenario == "lot4":
            main_lot4()
        if scenario == "lot5_10":
            main_lot5_10()
        if scenario == "lot1_lot4_review_beni":
            main_reannot_lot1_4_beni()
        if scenario == "lot1_lot4_review_sed":
            main_reannot_lot1_4_sed()
        else:
            ValueError("Wrong scenario")

    def generate_annotated_dataset(self, config_path):
        config = OmegaConf.load(config_path)
        logger.info(f"Config : {config}")
        for data_subset in os.listdir(config.DATA.COMPOSITE):
            self.generate_annotated_subset(config_path, data_subset)

    def generate_annotated_subset(self, config_path, data_subset):
        config = OmegaConf.load(config_path)
        logger.info(f"Config : {config}")
        ds_path = os.path.join(str(config.DATA.COMPOSITE), data_subset)
        mask_path = os.path.join(str(config.DATA.ANNOTATED_MASKS), data_subset)
        dataset_export_path = os.path.join(str(config.DATA.ANNOTATED_DATASET), data_subset)
        logger.info(f"generate_annotated_dataset for {data_subset}")
        generate_annotated_dataset(ds_path, mask_path, dataset_export_path)


if __name__ == '__main__':
    fire.Fire(Pipeline)

