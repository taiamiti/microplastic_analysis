
"""
# todo
first install clearml for local python using managed clearml server:
- pip install clearml[gs]
- edit ~/clearml.conf with google storage credential and bucket (add one for each project)


Then convert all scripts into clearml tasks
- Add the 2 magic lines
- Use clearml dataset and artefacts directly into those tasks
- Add logger
Run locally (not remotely with a remote agent)

Write a pipeline to connect these tasks

Add orchestration
- Add an extra task - add a scheduler for creating new clearml dataset version if new files or changes are found (use sync dataset)
- Add a trigger if a new clearml dataset is created to trigger the rest of the pipeline / tasks

"""
from loguru import logger

from omegaconf import OmegaConf
from pathlib import Path

from src.ingest_data import main as ingest_data
from src.create_composite import create_and_export_comp_dataset
from src.create_tasks import main as create_tasks


def pipe(config_path):
    config = OmegaConf.load(config_path)
    logger.info(f"Config : {config}")
    logger.info(f"Ingest data from {config.DATA.RAW.local_path}")
    ingest_data(config.DATA.RAW.local_path,
                config.LOCAL_EMBEDDING_CENTER.local_path,
                config.DATA.INGESTED.local_path)
    logger.info(f"create_and_export_comp_dataset from {config.DATA.INGESTED.local_path}")
    create_and_export_comp_dataset(config.DATA.INGESTED.local_path, config.DATA.COMPOSITE.local_path)
    # logger.info(f"create_tasks from {config.DATA.COMPOSITE.local_path}")
    # create_tasks(config.DATA.COMPOSITE.local_path,
    #              config.DATA.TASKS.local_path,
    #              config.CREATE_TASKS.n_clusters)


if __name__ == '__main__':
    # for cfg_path in sorted(list(Path('./configs').glob("config_lot1*.yaml"))):
    #     pipe(str(cfg_path))
    # pipe("./configs/config_lot4-28-06-2023-sediments-part3.yaml")
    # pipe("./configs/config_unit_test.yaml")
    # pipe("./configs/config_lot5-04-07-2023-benitiers-part1.yaml")
    # pipe("./configs/config_lot5-04-07-2023-benitiers-part2.yaml")
    # pipe("./configs/config_lot6-12-08-2023-eau-vertical.yaml")
    # pipe("./configs/config_lot6-12-08-2023-eau-horizontal.yaml")
    # pipe("./configs/config_lot2-30-05-2023-tak_nai-part2.yaml")
    # pipe("./configs/config_lot7-28-09-2023-benitiers.yaml")
    # pipe("./configs/config_lot8-28-09-2023-benitiers.yaml")
    # pipe("./configs/config_lot9-09-10-2023-benitiers.yaml")
    # pipe("./configs/config_lot10-09-10-2023-benitiers.yaml")
    pipe("./configs/config_lot11-20-11-2023-eau.yaml")
