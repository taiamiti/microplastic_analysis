export PYTHONPATH=$PWD

# define var
lot=lot1-20-04-2023-benitiers

# ingest data
python src/pipeline.py ingest_data_subset configs/test_config_lot1.yaml $lot
# create composite
python src/pipeline.py create_composite_subset configs/test_config_lot1.yaml $lot

# create task
python src/pipeline.py create_tasks configs/test_config_lot1.yaml $lot

# copy labkit generated annotations to annotated_data (manual op)
# copy matching names
python src/pipeline.py matching_old_names_with_new configs/test_config_lot1.yaml $lot

# generate annotated subset
python src/pipeline.py generate_annotated_subset configs/test_config_lot1.yaml lot1-20-04-2023-benitiers
