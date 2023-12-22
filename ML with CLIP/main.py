from io import BytesIO
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import requests
import torch
from PIL import Image
import os
import clip
import matplotlib.pyplot as plt
from umap import UMAP
import hdbscan
from tqdm import tqdm
import ssl
import pandas as pd
import ast

ssl._create_default_https_context = ssl._create_unverified_context

folder_path = 'dataset'
output_file = 'all_urls.txt'

with open(output_file, 'w') as f:
    file_list = os.listdir(folder_path)

    for file_name in file_list:
        if file_name.endswith('.csv'):
            file_path = os.path.join(folder_path, file_name)

            df = pd.read_csv(file_path)

            df['sizes'] = df['sizes'].apply(ast.literal_eval)

            condition = (df['sizes'].apply(lambda x: any(item['type'] == 'x' for item in x))) & (df['album_id'] == -6)
            url_x = df.loc[condition, 'sizes'].apply(
                lambda x: next(item['url'] for item in x if item['type'] == 'x'))

            url_x.to_csv(f, header=False, index=False)


output_folder = 'final_clusters'
os.makedirs(output_folder, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

with open(output_file, 'r') as f:
    urls = f.read().splitlines()


def process_image(url):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            image = Image.open(BytesIO(response.content)).convert("RGB")
            image_input = preprocess(image).unsqueeze(0).to(device)
            with torch.no_grad():
                image_feature = model.encode_image(image_input)
            return image_feature.cpu().numpy()
    except Exception as e:
        pass


max_threads = 8

with ThreadPoolExecutor(max_threads) as executor:
    image_features = list(tqdm(executor.map(process_image, urls), total=len(urls), desc="Processing images"))

valid_image_features = [arr for arr in image_features if arr is not None and arr.shape[1] == 512]
image_features = np.vstack(valid_image_features)

dimension_model = UMAP(n_neighbors=5,
                       n_epochs=150,
                       min_dist=0.09,
                       n_components=2,
                       )

train_clusterable_embedding = dimension_model.fit_transform(image_features)

cluster_model = hdbscan.HDBSCAN(min_cluster_size=20, min_samples=10, cluster_selection_method = 'leaf')

train_labels = cluster_model.fit_predict(train_clusterable_embedding)

cluster_info = {}
for idx, label in enumerate(train_labels):
    if label != -1:
        if label not in cluster_info:
            cluster_info[label] = []
        cluster_info[label].append(idx)


def save_links(url_idx_tuple, links_file):
    url, idx = url_idx_tuple
    try:
        if url.strip():
            with open(links_file, 'a') as f:
                f.write(f"{url}\n")
    except Exception as e:
        pass



max_threads_download = 8

with ThreadPoolExecutor(max_threads_download) as executor:
    tasks = []
    for cluster, elements in cluster_info.items():
        cluster_folder = os.path.join(output_folder, f"cluster_{cluster}")
        os.makedirs(cluster_folder, exist_ok=True)

        urls_to_save = [urls[idx] for idx in elements]
        with open(os.path.join(cluster_folder, f"urls_cluster_{cluster}.txt"), 'w') as f:
            f.write('\n'.join(urls_to_save))

            for url_idx in tqdm(elements, desc=f"Saving URLs for Cluster {cluster}"):
                tasks.append((urls[url_idx], url_idx))


plt.figure(figsize=(10, 7))
for cluster, elements in cluster_info.items():
    if cluster != -1:
        cluster_points = train_clusterable_embedding[elements]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {cluster}')

plt.title('HDBSCAN Clustering Visualization')
plt.xlabel('UMAP Feature 1')
plt.ylabel('UMAP Feature 2')
plt.show()