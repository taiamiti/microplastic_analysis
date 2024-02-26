export PYTHONPATH=$PWD

# define var
lot=lot1-20-04-2023-benitiers

# ingest data
python src/pipeline.py ingest_data_subset configs/test_config_lot1.yaml $lot
# create composite
python src/pipeline.py create_composite_subset configs/test_config_lot1.yaml $lot

# create task
python src/pipeline.py create_tasks configs/test_config_lot1.yaml new_lot