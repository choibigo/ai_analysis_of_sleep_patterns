import pandas as pd
from sklearn.metrics import silhouette_score
from sklearn.metrics import silhouette_samples
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from pathlib import Path

def get_data(): # data 폴더에 있는 clustering_data.csv 파일을 읽어서 반환
    util_file_dir = Path(__file__).resolve()
    data_dir = util_file_dir.parents[2]/'data'
    data_csv = pd.read_csv(data_dir/'clustering_data.csv')
    data = data_csv.values
    return data

def regularize_data(data, method): # data를 method에 따라 정규화
    '''
    method: 'StandardScaler' or 'Std', 'MinMaxScaler' or 'MM
    '''
    if method == 'StandardScaler' or method == 'Std':
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
    elif method == 'MinMaxScaler' or method == 'MM':
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
    else:
        raise ValueError('Invalid method')
    return scaler.fit_transform(data)

def grid_search_kmeans(Data, min_cluster_num, max_cluster_num): # KMeans의 최적 클러스터 수 찾기
    util_file_dir = Path(__file__).resolve()
    n_clusters = np.arange(min_cluster_num, max_cluster_num)
    silhouette = np.zeros(len(n_clusters))

    for cluster_index, n_cluster in enumerate(n_clusters):
        kmeans = KMeans(n_clusters=n_cluster, random_state=0, n_init='auto')
        kmeans.fit(Data)
        labels = kmeans.labels_
        silhouette[cluster_index] = silhouette_score(Data, labels)
        if np.max(silhouette) == silhouette[cluster_index]:
            max_silhouette = silhouette[cluster_index]
            best_n_clusters = n_cluster
            best_labels = labels
    with open(util_file_dir.parents[0]/'Results/Average_Silhouette_Score_KMeans.txt', 'w') as f:
        f.write(f'Best number of clusters: {best_n_clusters}\n')
        f.write(f'Average silhouette score: {max_silhouette}\n')
    return best_n_clusters, best_labels, max_silhouette

def grid_search_dbscan(Data, min_eps, max_eps, min_sample_num, max_sample_num): # DBSCAN의 최적 클러스터 수 찾기
    util_file_dir = Path(__file__).resolve()
    eps = np.linspace(min_eps, max_eps, 10)
    min_samples = np.linspace(min_sample_num, max_sample_num, max_sample_num-min_sample_num+1, dtype=int)
    silhouette = np.zeros((len(eps), len(min_samples)))

    for eps_index, target_eps in enumerate(eps):
        for sample_index, target_sample_num in enumerate(min_samples):
            dbscan = DBSCAN(eps=target_eps, min_samples=target_sample_num)
            labels = dbscan.fit_predict(Data)
            unique_labels = np.unique(labels)
            unique_label_len = len(unique_labels) - (1 if -1 in labels else 0)
            if unique_label_len>1:
                silhouette[eps_index, sample_index] = silhouette_score(Data, labels)
                if np.max(silhouette) == silhouette[eps_index, sample_index]:
                    max_silhouette = silhouette[eps_index, sample_index]
                    best_labels = labels
                    best_n_clusters = len(np.unique(labels))
                    print(f'eps: {target_eps}, min_samples: {target_sample_num}, silhouette: {max_silhouette}')
    with open(util_file_dir.parents[0]/'Results/Average_Silhouette_Score_DBSCAN.txt', 'w') as f:
        f.write(f'Best number of clusters: {best_n_clusters}\n')
        f.write(f'Average silhouette score: {max_silhouette}\n')

    return best_n_clusters, best_labels, max_silhouette

def figure_silhouette(Data, labels, algorithm): # silhouette plot 그리기
    util_file_dir = Path(__file__).resolve()
    silhouette_avg = silhouette_score(Data, labels)
    silhouette_coefficients = silhouette_samples(Data, labels)
    unique_labels = set(labels)
    n_clusters = len(unique_labels) - (1 if -1 in labels else 0)
    plt.figure(figsize=(10, 7))

    if n_clusters > 1:
        y_lower = 10
        for i in unique_labels:
            if i == -1:
                continue

            ith_cluster_silhouette_values = silhouette_coefficients[labels == i]
            ith_cluster_silhouette_values.sort()

            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i

            color = plt.cm.Spectral(float(i) / n_clusters)
            plt.fill_betweenx(np.arange(y_lower, y_upper),
                              0, ith_cluster_silhouette_values,
                              facecolor=color, edgecolor=color, alpha=0.7)

            plt.text(-0.05, y_lower + 0.5 * size_cluster_i, f'Label {i}')

            y_lower = y_upper + 10

        plt.axvline(x=silhouette_avg, color="red", linestyle="--")

    plt.title(f"Silhouette Scores of {algorithm} for {n_clusters} Clusters")
    plt.xlabel("Silhouette coefficient values")
    plt.ylabel("Cluster labels")
    plt.yticks([])
    plt.savefig(util_file_dir.parents[0] / 'Results' / f'Silhouette_Plot_{algorithm}.png')



def figure_manifold(Data, labels, method, algorithm): # manifold plot 그리기
    util_file_dir = Path(__file__).resolve()
    unique_labels = np.unique(labels)
    if len(unique_labels) == 2:
        cmap = ListedColormap(['blue', 'pink'])
        plt.figure(figsize=(8, 6))
        for label in np.unique(labels):
            mask = (labels == label)
            plt.scatter(Data[mask, 0], Data[mask, 1], color=cmap(label), s=10, label=f'Label {label}', alpha=0.8)
        plt.title(f'{method} Visualization with Labels by {algorithm}')
        plt.xlabel(f'{method} Component 1')
        plt.ylabel(f'{method} Component 2')
        plt.legend()
    else:
        plt.figure(figsize=(8, 6))
        for label in np.unique(labels):
            mask = (labels == label)
            plt.scatter(Data[mask, 0], Data[mask, 1], s=10, label=f'Label {label}', alpha=0.8)
        plt.title(f'{method} Visualization with Labels by {algorithm}')
        plt.xlabel(f'{method} Component 1')
        plt.ylabel(f'{method} Component 2')
        plt.legend()
    plt.savefig(util_file_dir.parents[0] / 'Results' / f'{method}_{algorithm}.png')