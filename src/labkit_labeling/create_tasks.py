import os

import fire
import numpy as np
import fiftyone.brain as fob
import fiftyone as fo
from fiftyone import ViewField as F

from sklearn.cluster import KMeans, DBSCAN
import plotly.express as px
from sklearn.model_selection import train_test_split

CLUSTER_FIELD_NAME = "cluster"


def cluster_dataset(dataset: fo.Dataset, n_clusters=15) -> fo.Dataset:
    embeddings = np.array(dataset.values("embeddings"))
    kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init="auto").fit(embeddings)
    cluster_ids = kmeans.predict(embeddings)
    dataset.set_values(CLUSTER_FIELD_NAME, cluster_ids)
    return dataset


def cluster_dataset_dbscan(dataset: fo.Dataset, eps=0.5, min_samples=5) -> fo.Dataset:
    embeddings = np.array(dataset.values("embeddings"))
    dbscan = DBSCAN(eps=eps, min_samples=min_samples).fit(embeddings)
    cluster_ids = [int(label) + 1 for label in dbscan.labels_]  # -1 is used for noise
    dataset.set_values(CLUSTER_FIELD_NAME, cluster_ids)
    return dataset


def visualize_clusters(dataset):
    embeddings = np.array(dataset.values("embeddings"))

    # Compute 2D representation using pre-computed embeddings
    results = fob.compute_visualization(
        dataset,
        embeddings=embeddings,
        num_dims=2,
        brain_key="image_embeddings",
        verbose=True,
        seed=51,
    )
    # Visualize image embeddings colored by cluster
    scatterplot_fig = plot_embeddings_scatterplot(results)
    # Visualize num images per clusters
    histo_fig = plot_cluster_histogram(dataset)
    return scatterplot_fig, histo_fig


def plot_embeddings_scatterplot(results):
    scatterplot_fig = results.visualize(
        axis_equal=True,
        labels=CLUSTER_FIELD_NAME
    )
    # scatterplot_fig.show(height=512)
    return scatterplot_fig._figure  # convert interactivescatterplot from fiftyone to regular plotly figure


def plot_cluster_histogram(dataset):
    cluster_ids = dataset.values(CLUSTER_FIELD_NAME)
    counts, bins = np.histogram(cluster_ids, bins=np.unique(cluster_ids))
    bins = 0.5 * (bins[:-1] + bins[1:])
    fig = px.bar(x=bins, y=counts, labels={'x': 'cluster_id', 'y': 'count'})
    # fig.show()
    return fig


def export_tasks(dataset, output_dir, max_samples=20):
    # set max samples per cluster
    cluster_ids = dataset.values(CLUSTER_FIELD_NAME)
    for cluster_id in np.unique(cluster_ids).astype(int).tolist():
        if max_samples == -1:
            sel_view = dataset.match(F(CLUSTER_FIELD_NAME) == cluster_id)
        else:
            sel_view = dataset.match(F(CLUSTER_FIELD_NAME) == cluster_id).take(max_samples)
        # Export the view
        sel_view.export(
            export_dir=os.path.join(output_dir, "{:02d}").format(int(cluster_id)),
            dataset_type=fo.types.ImageDirectory,
            export_media=True
        )


def sample_dataset_by_origin(dataset, sampling_rate=0.25):
    if sampling_rate == 1:
        return dataset
    origin = ["-".join(l) for l in zip(dataset.values('sample_type'),
                                       dataset.values('island'),
                                       dataset.values('station'),
                                       dataset.values('replica'),
                                       )]
    dataset.set_values('origin', origin)
    sample_ids = dataset.values('id')
    origins = dataset.values('origin')
    sid_train, sid_test, ori_train, ori_test = train_test_split(sample_ids,
                                                                origins,
                                                                stratify=origins,
                                                                test_size=sampling_rate,
                                                                random_state=42
                                                                )
    data_to_annotate = dataset[sid_test]
    return data_to_annotate


def sample_dataset_by_group(dataset, sampling_rate=0.25):
    if sampling_rate == 1:
        return dataset
    sample_ids = dataset.values('id')
    groups = np.array(dataset.values(CLUSTER_FIELD_NAME))
    group_ids, counts = np.unique(groups, return_counts=True)
    singleton_groups = []
    valid_groups = []
    # handle edge case where singleton groups exist due to outlier which is not possible to split afterwards
    for group_id, count in zip(group_ids, counts):
        if count < 2:
            singleton_groups.append(group_id)
        else:
            valid_groups.append(group_id)
    for singleton_group in singleton_groups:
        groups[groups == singleton_group] = valid_groups[0]  # assign singleton groups to first valid group
    sid_train, sid_test, ori_train, ori_test = train_test_split(sample_ids,
                                                                groups,
                                                                stratify=groups,
                                                                test_size=sampling_rate,
                                                                random_state=42
                                                                )
    data_to_annotate = dataset[sid_test]
    return data_to_annotate


def create_tasks_subset(input_dir, output_dir, n_clusters, max_samples, save_plot=True):
    dataset = fo.Dataset.from_dir(
        dataset_type=fo.types.FiftyOneDataset,
        dataset_dir=input_dir,
        labels_path=None,
    )
    dataset = cluster_dataset(dataset, n_clusters=n_clusters)
    if save_plot:
        scatterplot_fig, histo_fig = visualize_clusters(dataset)
        scatterplot_fig.write_html(os.path.join(output_dir, "scatterplot_fig.html"))
        histo_fig.write_html(os.path.join(output_dir, "histo_fig.html"))
    export_tasks(dataset, output_dir, max_samples)


def create_tasks_with_sampling(input_dir, lots, output_dir, sampling_type,
                               sampling_rate=0.25, n_clusters=15, max_samples=20, save_plot=True):
    # Merge to single dataset
    dataset = fo.Dataset()
    for lot in lots:
        dataset.add_dir(
            dataset_dir=os.path.join(input_dir, lot),
            dataset_type=fo.types.FiftyOneDataset,
        )

    if sampling_type == "origin":
        dataset = sample_dataset_by_origin(dataset, sampling_rate=sampling_rate)
        dataset = cluster_dataset(dataset, n_clusters=n_clusters)  # cluster after sampling with less clusters
    if sampling_type == "group":
        dataset = cluster_dataset(dataset, n_clusters=n_clusters)  # cluster before sampling with many clusters
        dataset = sample_dataset_by_group(dataset, sampling_rate=sampling_rate)
    else:
        raise ValueError("sampling type is invalid")
    if save_plot:
        os.makedirs(output_dir, exist_ok=True)
        scatterplot_fig, histo_fig = visualize_clusters(dataset)
        scatterplot_fig.write_html(os.path.join(output_dir, "scatterplot_fig.html"))
        histo_fig.write_html(os.path.join(output_dir, "histo_fig.html"))
    export_tasks(dataset, output_dir, max_samples)


def create_tasks_lot5_10(input_dir, output_dir):
    lots = [
        "lot2-30-05-2023-tak_nai-part2",
        "lot5-04-07-2023-benitiers-part1",
        "lot5-04-07-2023-benitiers-part2",
        "lot6-12-08-2023-eau-vertical",
        "lot6-12-08-2023-eau-horizontal",
        "lot7-28-09-2023-benitiers",
        "lot8-28-09-2023-benitiers",
        "lot9-09-10-2023-benitiers",
        "lot10-09-10-2023-benitiers"
    ]
    create_tasks_with_sampling(input_dir,
                               lots,
                               output_dir,
                               "group",
                               sampling_rate=0.25,
                               n_clusters=30,
                               max_samples=20,
                               save_plot=True)


def create_tasks_lot1_3(input_dir, output_dir):
    lots = [
        "lot1-20-04-2023-benitiers",
        "lot1-20-04-2023-sediments",
        "lot2-30-05-2023-tak_nacl",
        "lot2-30-05-2023-tak_nai",
        "lot3-08-06-2023-benitiers"
    ]
    create_tasks_with_sampling(input_dir,
                               lots,
                               output_dir,
                               "group",
                               sampling_rate=0.25,
                               n_clusters=15,
                               max_samples=20,
                               save_plot=True)


def create_tasks_lot4(input_dir, output_dir):
    lots = [
        "lot4-28-06-2023-sediments-part1",
        "lot4-28-06-2023-sediments-part2",
        "lot4-28-06-2023-sediments-part3",
    ]
    create_tasks_with_sampling(input_dir,
                               lots,
                               output_dir,
                               "origin",
                               sampling_rate=0.25,
                               n_clusters=15,
                               max_samples=-1,
                               save_plot=True)


if __name__ == '__main__':
    fire.Fire(create_tasks_subset)


