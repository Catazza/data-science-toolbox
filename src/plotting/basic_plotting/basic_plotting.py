import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from typing import Tuple
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale


def create_box_plot(data: pd.DataFrame,
                    plot_var: str,
                    grouping_var: str,
                    figsize: Tuple=(16, 16),
                    whis: float=1.5,
                    outlier_marker: Tuple=('red', 5),
                    fontsize: int=12,
                    rotation: int=90) -> None:
    """
    Function to create a box plot with sensible defaults and chart titles and labels.

    :param data: data in the form of pandas df
    :param plot_var: variable onto which to create the boxplot
    :param grouping_var: column indicating the groups
    :param figsize:
    :param whis:
    :param outlier_marker:
    :param fontsize:
    :param rotation:
    :return:
    """
    outlier_marker = dict(markerfacecolor=outlier_marker[0], markersize=outlier_marker[1])
    plt.figure(figsize=figsize)
    sns.boxplot(x=grouping_var, y=plot_var, data=data, palette="Set3", whis=whis,
                flierprops=outlier_marker)
    plt.xticks(rotation=rotation)
    plt.xlabel(grouping_var, fontsize=fontsize)
    plt.ylabel(plot_var, fontsize=fontsize)
    plt.tight_layout()


def grouped_bar_plot(data: pd.DataFrame,
                     target_var: str,
                     grouping_var: str,
                     agg_method: str,
                     target_var_label: str='target var',
                     grouping_var_label: str='grouping var',
                     ) -> None:
    """
    Function to show the sum or count for groups within the data, in a bar chart format.

    :param data:
    :param target_var:
    :param grouping_var:
    :param agg_method:
    :param target_var_label:
    :param grouping_var_label:
    :return:
    """
    data_to_plot = data.groupby(grouping_var).agg({target_var: agg_method}).reset_index()
    sns.barplot(x=grouping_var, y=target_var, color='blue', data=data_to_plot, ci=0)
    plt.xlabel(grouping_var_label, fontsize=12)
    plt.ylabel(target_var_label, fontsize=12)
    plt.tight_layout()
    plt.show()


def coloured_scatter_plot(data: pd.DataFrame,
                          label_col: str,
                          x_axis_col: str,
                          y_axis_col: str) -> None:
    """
    Creates a scatter plot, with each point being colored according to the belonging to a category, defined in a column
    of the data.

    :param data:
    :param label_col:
    :param x_axis_col:
    :param y_axis_col:
    :return:
    """
    category_vals = data[label_col].sort_values().unique()
    fg = sns.FacetGrid(data=data, hue=label_col, hue_order=category_vals, legend_out=False)
    fg.map(plt.scatter, x_axis_col, y_axis_col).add_legend()
    plt.tight_layout()
    plt.show()


def pca_explained_variance_plot(data: pd.DataFrame,
                                n_components: int,
                                do_scale: bool = True) -> None:
    """
    Creates 2 plots:
        - explained variance for each component
        - cumulative variance by each component

    :param data:
    :param n_components:
    :param do_scale:
    :return:
    """
    # First, perform the PCA reduction
    _, pca = _perform_pca(data, n_components, do_scale)

    # Explained variance by component
    explained = np.round(pca.explained_variance_ratio_, decimals=4) * 100
    print("<< Explained variance by component: ", list(explained))
    plt.figure(figsize=(16, 16))
    plt.barh(y=list(range(len(explained))), width=list(explained))
    plt.title('Variance explained by each Principal Component')

    # Cumulative Variance
    cum_var = np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4) * 100)
    cum_var = [0] + list(cum_var)
    print("<< Cumulative explained variance", cum_var)
    plt.figure(figsize=(16, 16))
    plt.plot(cum_var)
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)
    plt.title('Cumulative variance explained')


def _perform_pca(data: pd.DataFrame,
                 n_components: int,
                 do_scale: bool=True):
    if do_scale:
        data = scale(data)
    pca = PCA(n_components=n_components, svd_solver='full')
    pca.fit(data)
    return pca.transform(data), pca


def pca_biplot(data: pd.DataFrame,
               do_scale: bool=True,
               plot_arrows: bool=False):
    """
    Function to create a PCA biplot.

    :param data:
    :param do_scale:
    :param plot_arrows:
    :return:
    """
    # First, perform PCA
    transformed, pca = _perform_pca(data, 2, do_scale)

    # Then plot
    reduced_x = transformed[:, 0]
    reduced_y = transformed[:, 1]
    plt.figure(figsize=(16, 16))
    plt.scatter(transformed[:, 0], transformed[:, 1], s=10, marker='o', edgecolors='face')
    plt.title('PCA biplot')

    if plot_arrows:
        x_vector = pca.components_[0]
        y_vector = pca.components_[1]
        # Creating arrows for the biplot
        for i in range(len(x_vector)):
            plt.arrow(0, 0, x_vector[i] * max(reduced_x), y_vector[i] * max(reduced_y),
                      color='r', width=0.0005, head_width=0.0025)
            plt.text(x_vector[i] * max(reduced_x) * 1.2, y_vector[i] * max(reduced_y) * 1.2, data.columns.tolist()[i],
                     color='r')
    plt.axhline(0, color='gray')  # add horizontal axis at 0
    plt.axvline(0, color='gray')  # add vertical axis at 0
