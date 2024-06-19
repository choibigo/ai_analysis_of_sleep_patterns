import utils
from sklearn.manifold import TSNE
from matplotlib.colors import ListedColormap
import umap
import time

# 데이터 불러오기
Raw_Data = utils.get_data()
print('Data Read Complete')
print(f'Total Data Num: {Raw_Data.shape[0]}')
print(f'Feature Num: {Raw_Data.shape[1]}\n')

# 데이터 정규화
MM_Data = utils.regularize_data(Raw_Data, 'StandardScaler')

# 클러스터링
Algorithm = 'KMeans'
print(f'Grid Searching {Algorithm}...')
t1 = time.time()
if Algorithm == 'KMeans':
    best_n_clusters, labels, max_silhouette = utils.grid_search_kmeans(MM_Data, 2, 10)
elif Algorithm == 'DBSCAN':
    best_n_clusters, labels, max_silhouette = utils.grid_search_dbscan(MM_Data, 0.1, 1, 2, 10)
elapsed_time = time.time()-t1
print(f'Grid Search finished: {elapsed_time:.3f}s \n')

# 실루엣 계수 시각화
utils.figure_silhouette(MM_Data, labels, Algorithm)

# t-SNE, UMAP -> Results에 시각화 결과 저장
print('Starting T-SNE...')
t1 = time.time()
tsne = TSNE(n_components=2,random_state=0)
Data_tsne = tsne.fit_transform(MM_Data)
elapsed_time = time.time()-t1
print(f'T-SNE finished: {elapsed_time:.3f}s \n')
utils.figure_manifold(Data_tsne, labels, 'T-SNE', Algorithm)

print('Starting UMAP...')
t1=time.time()
umap_model = umap.UMAP(n_components=2)
Data_umap = umap_model.fit_transform(MM_Data)
elapsed_time = time.time()-t1
print(f'UMAP finished: {elapsed_time:.3f}s\n')
utils.figure_manifold(Data_umap, labels, 'UMAP', Algorithm)