import fire
from loguru import logger

from omegaconf import OmegaConf
from src.data_prep.ingest_data import main as ingest_data_fun
from src.data_prep.create_composite import create_and_export_comp_dataset
import os

from src.labkit_labeling.create_tasks import create_tasks_with_sampling
from src.labkit_labeling.matching_old_names_with_new import main_lot3, main_lot5_10, main_reannot_lot1_4_beni, \
    main_reannot_lot1_4_sed, main_lot4, copy_matching_name
from src.labkit_labeling.generate_annotated_dataset import main as generate_annotated_dataset
from src.labkit_labeling.prepare_dataset_for_openmmseg import main as prepare_dataset_for_openmmseg


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
            create_tasks_with_sampling(config.DATA.COMPOSITE,
                                       config.CREATE_TASKS_LOT4.lots,
                                       os.path.join(config.DATA.TO_ANNOTATE, "lot4"),
                                       config.CREATE_TASKS_LOT4.sampling_type,
                                       sampling_rate=config.CREATE_TASKS_LOT4.sampling_rate,
                                       n_clusters=config.CREATE_TASKS_LOT4.n_clusters,
                                       max_samples=config.CREATE_TASKS_LOT4.max_samples,
                                       save_plot=config.CREATE_TASKS_LOT4.save_plot)
        if scenario == "lot5_10":
            create_tasks_with_sampling(config.DATA.COMPOSITE,
                                       config.CREATE_TASKS_LOT5_10.lots,
                                       os.path.join(config.DATA.TO_ANNOTATE, "lot5_10"),
                                       config.CREATE_TASKS_LOT5_10.sampling_type,
                                       sampling_rate=config.CREATE_TASKS_LOT5_10.sampling_rate,
                                       n_clusters=config.CREATE_TASKS_LOT5_10.n_clusters,
                                       max_samples=config.CREATE_TASKS_LOT5_10.max_samples,
                                       save_plot=config.CREATE_TASKS_LOT5_10.save_plot)
        if scenario == "lot1_3":
            create_tasks_with_sampling(config.DATA.COMPOSITE,
                                       config.CREATE_TASKS_LOT1_3.lots,
                                       os.path.join(config.DATA.TO_ANNOTATE, "lot1_3"),
                                       config.CREATE_TASKS_LOT1_3.sampling_type,
                                       sampling_rate=config.CREATE_TASKS_LOT1_3.sampling_rate,
                                       n_clusters=config.CREATE_TASKS_LOT1_3.n_clusters,
                                       max_samples=config.CREATE_TASKS_LOT1_3.max_samples,
                                       save_plot=config.CREATE_TASKS_LOT1_3.save_plot)
        else:
            lots = [scenario]
            create_tasks_with_sampling(config.DATA.COMPOSITE,
                                       lots,
                                       os.path.join(config.DATA.TO_ANNOTATE, scenario),
                                       config.CREATE_TASKS.sampling_type,
                                       sampling_rate=config.CREATE_TASKS.sampling_rate,
                                       n_clusters=config.CREATE_TASKS.n_clusters,
                                       max_samples=config.CREATE_TASKS.max_samples,
                                       save_plot=config.CREATE_TASKS.save_plot)

    def matching_old_names_with_new(self, config_path, scenario):
        config = OmegaConf.load(config_path)
        logger.info(f"Config : {config}")
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
            dataset_dir = os.path.join(config.DATA.COMPOSITE, scenario, "data")
            annotated_dir = os.path.join(config.DATA.ANNOTATED_TMP, scenario)
            new_annotated_dir = os.path.join(config.DATA.ANNOTATED_MASKS, scenario)
            copy_matching_name(dataset_dir, annotated_dir, new_annotated_dir)

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

    def prepare_dataset_for_openmmseg(self, config_path):
        config = OmegaConf.load(config_path)
        logger.info(f"Config : {config}")
        ds_path = str(config.DATA.ANNOTATED_DATASET)
        save_dir = str(config.DATA.TRAINVAL_DATASET)
        prepare_dataset_for_openmmseg(ds_path, save_dir)


if __name__ == '__main__':
    fire.Fire(Pipeline)

