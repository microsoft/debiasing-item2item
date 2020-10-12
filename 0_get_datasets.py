#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import requests
import zipfile
import os
import lib

# Similarity labels -- see https://conservancy.umn.edu/handle/11299/198736
LABELED_PAIRS = "https://conservancy.umn.edu/bitstream/handle/11299/198736/pair-responses.csv?sequence=9&isAllowed=y"
ML25M_URL = "http://files.grouplens.org/datasets/movielens/ml-25m.zip"
ML25M_FILE = "datasets/ml-25m.zip"


def load_movielens():
    os.makedirs("datasets/", exist_ok=True)

    if not os.path.isfile(ML25M_FILE):
        print("Downloading MovieLens dataset ...")
        movielens_f = requests.get(ML25M_URL, stream=True)

        with open(ML25M_FILE, 'wb') as downloaded_file:
            for chunk in movielens_f.iter_content(chunk_size=1024*1024):
                downloaded_file.write(chunk)

        with zipfile.ZipFile(ML25M_FILE) as myzip:
            myzip.extractall("datasets")

    if not os.path.isdir("datasets/ml-25m-implicit"):
        print("Converting to implicit dataset ...")
        lib.convert_to_implicit("ml-25m")

    print("Reading MovieLens dataset ...")
    with zipfile.ZipFile(ML25M_FILE) as myzip:
        with myzip.open('ml-25m/ratings.csv') as ratings_f:
            ratings = pd.read_csv(ratings_f)
        with myzip.open('ml-25m/movies.csv') as movies_f:
            movies = pd.read_csv(movies_f)
    return ratings, movies


def get_reliable_responses(responses, ratings, movies):
    top_2500_ids = ratings.movieId.value_counts().index[:2500].tolist()
    inventory_ids = set(movies.movieId.tolist())
    responses["seedValid"] = responses.movieId.isin(top_2500_ids)
    responses["neighborValid"] = responses.neighborId.isin(inventory_ids)
    responses["valid"] = responses["seedValid"] & responses["neighborValid"]

    reliable_responses = responses[responses.valid].groupby(["movieId", "neighborId"]).filter(lambda x: len(x) >= 2)[
        ["movieId", "neighborId", "sim", "goodRec"]]
    return reliable_responses.groupby(["movieId", "neighborId"]).mean().sort_values(by=['movieId', 'sim'], ascending=[True, False])


def remove_rankings_with_not_enough_relevant_movies(reliable_responses):
    '''remove rankings without at least five relevant candidate movies'''
    def bin_by_relevance(x):
        bins = [-1, 2.0,  5]
        x["sim_bin"] = pd.cut(x["sim"], bins, labels=[0, 1], right=False)
        x["goodRec_bin"] = pd.cut(x["goodRec"], bins, labels=[0, 1], right=False)
        x["valid_group"] = (x["sim_bin"] == 1).sum() > 4
        return x

    diverse_rankings = reliable_responses.groupby("movieId").apply(bin_by_relevance)
    print("Before filtering out empty rankings:",
          diverse_rankings.reset_index()["movieId"].nunique())
    diverse_rankings = diverse_rankings[diverse_rankings.valid_group]
    print("After filtering out non-degenerate rankings:",
          diverse_rankings.reset_index()["movieId"].nunique())
    return diverse_rankings.reset_index()


def split_into_train_val_test(df, path):
    np.random.seed(42)
    a_idx = np.random.permutation(df["movieId"].unique())
    df.to_csv(path.replace(".csv", ".all.csv"), index=False)

    train_size = int(round(len(a_idx)*0.5))
    val_size = int(round(len(a_idx)*0.25))
    splits = dict(zip(['train', 'dev', 'test'], np.split(a_idx, [train_size, train_size+val_size])))

    for name, idx in splits.items():
        print(name, len(idx), 'items')
        df[df['movieId'].isin(idx)].to_csv(path.replace(".csv", "." + name + ".csv"), index=False)


def main():
    responses = pd.read_csv(LABELED_PAIRS)
    ratings, movies = load_movielens()

    reliable_responses = get_reliable_responses(responses, ratings, movies)
    processed_rankings = remove_rankings_with_not_enough_relevant_movies(reliable_responses)

    os.makedirs("datasets/labeled/", exist_ok=True)
    split_into_train_val_test(processed_rankings, "datasets/labeled/similarity_judgements.csv")


if __name__ == '__main__':
    main()
