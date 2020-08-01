import glob
import gzip
import json
import os
import shutil
import zipfile
import pandas as pd
import torch
import urllib3
import requests
import logging
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
from typing import List
from pandas import DataFrame
from typing import Dict, Tuple, Sequence
from sklearn.preprocessing import KBinsDiscretizer
from Consts import *
from sklearn.cluster import KMeans
from clint.textui import progress

def strip_df(df) -> "DataFrame":
    cleaned_df = df.replace(r'\\r', ' ', regex=True)
    cleaned_df = cleaned_df.replace(r'\\n', ' ', regex=True)
    cleaned_df = cleaned_df.replace(r'[ ]+', ' ', regex=True)

    return cleaned_df


def load_csv_as_df(path) -> "DataFrame":
    df = pd.read_csv(path, index_col=False, header=None)

    return df


def create_csv_from_json(input_path, out_path):
    df = json.load(open(input_path, encoding='utf-8'))
    df = pd.DataFrame(df['tables'])
    df = df['rows']
    flat_list = [item for sublist in df for item in sublist]
    flat_list = [item for sublist in flat_list for item in sublist]
    df = pd.DataFrame(flat_list)
    df.to_csv(out_path, encoding='utf-8', index=False)


def compress_file(input_path, out_path=None):
    if (out_path == None):
        out_path = f'{input_path}.gz'

    assert out_path.endswith('.gz')
    with open(input_path, 'rb') as f_in:
        with open(out_path, 'wb') as f_out:
            with gzip.GzipFile(input_path, 'wb', fileobj=f_out) as f_out:
                shutil.copyfileobj(f_in, f_out)


def zip_folder(folder, out_file):
    assert out_file.endswith('.zip')
    zip_file = zipfile.ZipFile(out_file, 'w')
    for _, _, files in os.walk(folder):
        for file in files:
            if file.endswith('.json'):
                zip_file.write(out_file, file, compress_type=zipfile.ZIP_DEFLATED)
    zip_file.close()


def unzip_file(file, out_folder):
    assert file.endswith('.zip')
    os.makedirs(os.path.dirname(out_folder), exist_ok=True)
    zip_file = zipfile.ZipFile(file)
    zip_file.extractall(out_folder)
    zip_file.close()


def decompress_file(input_path, out_path=None):
    assert input_path.endswith('.gz')
    if (out_path == None):
        out_path = input_path[:-3]

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with gzip.open(input_path, 'rb') as f_in:
        with open(out_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)


def download_data(url, out_path):
    logging.info(f'Downloading contents from {url} to {out_path}')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    r = requests.get(url, stream=True)
    with open(out_path, 'wb') as f:
        total_length = int(r.headers.get('content-length'))
        for chunk in progress.bar(r.iter_content(chunk_size=1024), expected_size=(total_length / 1024) + 1):
            if chunk:
                f.write(chunk)
                f.flush()


def decompress_file(input_path, out_path=None):
    assert input_path.endswith('.gz')
    if (out_path == None):
        out_path = input_path[:-3]

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with gzip.open(input_path, 'rb') as f_in:
        with open(out_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)


def download_file(url, out_path):
    http = urllib3.PoolManager()
    r = http.request('GET', url, preload_content=False)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'wb') as out:
        while True:
            data = r.read(4096)
            if not data:
                break
            out.write(data)

    r.release_conn()


def compress_file(input_path, out_path=None):
    if (out_path == None):
        out_path = f'{input_path}.gz'

    assert out_path.endswith('.gz')
    with open(input_path, 'rb') as f_in:
        with open(out_path, 'wb') as f_out:
            with gzip.GzipFile(input_path, 'wb', fileobj=f_out) as f_out:
                shutil.copyfileobj(f_in, f_out)


def get_bucket_weights(values) -> List[float]:
    counter = Counter(values)
    majority = max(counter.values())

    return {cls: round(float(majority) / float(count), 2) for cls, count in counter.items()}.values()


def get_one_hot_tensor(nums, max_size) -> torch.Tensor:
    assert max(nums) < max_size

    t = torch.zeros((len(nums), max_size))
    torch.arange(len(nums))
    t[torch.arange(len(nums)), nums] = 1

    return t


def load_json_files_to_df(folder: str, df_columns: List[str]) -> "DataFrame":
    json_pattern = os.path.join(folder, '*.json')
    file_list = glob.glob(json_pattern)

    df_original = pd.DataFrame(data=None, columns=df_columns)
    for file in file_list:
        with open(file, encoding='utf-8') as json_file:
            json_data = json.load(json_file)
            tmp_df = pd.DataFrame(data=json_data, columns=df_columns)
            df_original = df_original.append(tmp_df, ignore_index=True)

    return df_original


def preprocess_kql_data(args, df: "DataFrame") -> Tuple["DataFrame", List[str], Dict[str, int], List[str], Dict[str, int], List[float]]:
    df[QUERY] = df[QUERY].str.replace('string_token', '[unused1]')
    df[DURATION] = pd.to_numeric(df[DURATION])
    df = df[df[DURATION].between(df[DURATION].quantile(.0), df[DURATION].quantile(args.outliers_threshold))]
    df = filter_out_small_containers(args, df)
    durations_np = df[[DURATION]].sort_values(by=DURATION, ascending=True).to_numpy()
    discretizer = KBinsDiscretizer(n_bins=args.buckets_count, encode='ordinal', strategy=args.buckets_strategy)
    labels = list(np.squeeze(discretizer.fit_transform(durations_np)))
    labels_set = set(labels)
    start = 0
    end = 0
    durations = list(np.squeeze(durations_np.astype(int)))
    intervals = []
    i = 0
    while i < len(labels):
        labels_set.remove(labels[i])
        while (i < len(labels) and not (labels[i] in labels_set)):
            end = durations[i]
            i += 1
        intervals += [(start, end)]
        start = end

    logging.info(f'intervals: {intervals}')
    bins = pd.IntervalIndex.from_tuples(intervals)
    df[DURATION_INTERVAL] = pd.cut(df[DURATION], bins)
    df[DURATION_BUCKET] = df[DURATION_INTERVAL].apply(lambda x: str(x))
    df = df.sort_values(by=[DURATION], ascending=True)
    df = df.dropna()

    bucket_names = df[DURATION_BUCKET].unique()
    bucket_to_idx = dict(zip(bucket_names, range(len(bucket_names))))

    container_names = df[CONTAINER].unique()
    container_to_idx = dict(zip(container_names, range(len(container_names))))

    bucket_weights = list(get_bucket_weights(df[DURATION_BUCKET].values))
    bucket_average_durations = sorted(list(df.groupby(DURATION_BUCKET)[DURATION].mean()))

    plt.figure(figsize=(50, 8))
    df_f = filter_out_small_buckets(max(int(0.1 * len(intervals)), 10), df)
    counts = df_f[DURATION_BUCKET].value_counts()[lambda x: x > 0]
    plt.bar(counts.index, counts.values)
    plt.savefig(f'{args.output_path}/buckets_histogram.png')
    plt.clf()

    return df, bucket_names, bucket_to_idx, container_names, container_to_idx, bucket_weights, bucket_average_durations


def filter_out_small_containers(args, df: "DataFrame") -> "DataFrame":
    df = df.copy()
    d = df.groupby([CONTAINER]).count().sort_values(by=[QUERY])
    chosen_container = d[d[QUERY] > args.minimal_queries_per_container]
    chosen_container = chosen_container.drop(columns=[DURATION, QUERY])
    df = pd.merge(df, chosen_container, left_on=CONTAINER, right_on=CONTAINER)

    return df


def filter_out_small_buckets(top, df: "DataFrame") -> "DataFrame":
    df_filtered = df.copy()
    top = df_filtered[DURATION_BUCKET].value_counts().sort_values(ascending=False).head(top)
    df_filtered = pd.DataFrame({DURATION_BUCKET: top.index}).merge(df_filtered, how='left')

    return df_filtered
