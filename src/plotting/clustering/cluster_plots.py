import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd
import numpy as np
from sklearn.metrics import silhouette_samples


def silhouette_score_samples(clustered_data: pd.DataFrame, llim: float=-0.1):
    """
    Function to show the silhouette score for each sample, segmented by cluster of belonging.

    :param clustered_data: pd.Dataframe, containing the transformed data that was used for the clustering and a column
    called 'cluster' that contains the cluster labels
    :paramm llim: Lower limit of the x-axis in the chart
    :return: None - produces a chart with the silhouette scores
    """
    # Create a subplot with 1 row and 2 columns
    fig, ax1 = plt.subplots(1, 1)
    fig.set_size_inches(16, 16)

    # The 1st subplot is the silhouette plot. the silhouette coefficient can range from -1, 1 - set the limit to the
    # desired lower bound
    ax1.set_xlim([llim, 1])
    # The (n_clusters+1)*10 is for inserting blank space between silhouette
    # plots of individual clusters, to demarcate them clearly.
    n_clusters = clustered_data['cluster'].nunique()
    ax1.set_ylim([0, len(clustered_data) + (n_clusters + 1) * 10])
    y_lower = 10
    # Calculate the silhouette sample scores
    data_cols = [col for col in clustered_data.columns.tolist() if col != 'cluster']
    clustered_data['silhouette_score'] = silhouette_samples(clustered_data[data_cols], clustered_data['cluster'])

    for i in range(n_clusters):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = \
            clustered_data[clustered_data['cluster'] == i]['silhouette_score'].values

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral(float(i) / n_clusters)
        ax1.fill_betweenx(np.arange(y_lower, y_upper),
                          0, ith_cluster_silhouette_values,
                          facecolor=color, edgecolor=color, alpha=0.7)

        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    ax1.set_title("The silhouette plot for the various clusters.")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")

    # The vertical line for average silhouette score of all the values
    ax1.axvline(x=clustered_data['silhouette_score'].mean(), color="red", linestyle="--")

    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

    plt.suptitle(("Silhouette analysis with n_clusters = %d, total observations of %d" % (
                      n_clusters, clustered_data.shape[0])),
                 fontsize=14, fontweight='bold')
    plt.show()


if __name__ == '__main__':
    from sklearn.datasets import make_blobs
    from sklearn.cluster import KMeans
    # Generating the sample data from make_blobs
    # This particular setting has one distinct cluster and 3 clusters placed close
    # together.
    X, y = make_blobs(n_samples=500,
                      n_features=2,
                      centers=4,
                      cluster_std=1,
                      center_box=(-10.0, 10.0),
                      shuffle=True,
                      random_state=1)  # For reproducibility

    data = pd.DataFrame(X)
    clusterer = KMeans(n_clusters=4, random_state=10)
    data['cluster'] = clusterer.fit_predict(X)
    silhouette_score_samples(data)
