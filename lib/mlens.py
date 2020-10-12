import scipy.sparse as sp
import pandas as pd
import numpy as np
import os
from pathlib import Path
import argparse
import collections
import json

PATH = "datasets/"


def from_json(file):
    with open(file, encoding='utf-8') as fh:
        return json.load(fh)


def to_json(file, data):
    os.makedirs(os.path.dirname(file), exist_ok=True)
    with open(file, 'w', encoding='utf8') as json_file:
        json.dump(data, json_file, ensure_ascii=False)


def ml(x, fpath=PATH):
    f_tags = fpath + "ml-25m/genome-scores.csv"
    prefix = fpath + x + "/"

    return (prefix + "ratings.csv", prefix + "movies.csv", f_tags)


def read_movielens_ratings(movielens_file):
    params = {"skiprows": 1, "engine": "c", "header": None}
    ratings_raw = pd.read_csv(movielens_file, names=[
                              "userId", "movieId", "rating", "timestamp"], **params)
    print("I have read %d lines " % len(ratings_raw))

    i = ratings_raw["movieId"].values
    j = ratings_raw["userId"].values

    ratings = sp.coo_matrix((ratings_raw["rating"].values, (i, j))).transpose()
    timestamps = sp.coo_matrix((ratings_raw["timestamp"].values, (i, j))).transpose()

    return ratings, timestamps


class ImplicitMovieLens(object):
    def __init__(self, version):
        f_ratings, f_titles, f_tags = ml(version)
        f_ratings_npz = ".".join(f_ratings.split(".")[:-1] + ["npz"])

        if os.path.exists(f_ratings_npz):
            with np.load(f_ratings_npz) as f:
                ratings = f["ratings"]
            i = ratings[:, 0]
            j = ratings[:, 1]
            data = ratings[:, 2]
        else:
            ratings_raw = pd.read_csv(f_ratings, names=["userId", "movieId", "rating"], header=None)
            i = ratings_raw["userId"].values
            j = ratings_raw["movieId"].values
            data = (ratings_raw["rating"]).astype(int)

        ratings = sp.coo_matrix((data, (i, j))).tocsc()
        print("Read %d x %d user-item matrix" % (ratings.shape[0], ratings.shape[1]))

        items = pd.read_csv(f_titles, index_col="newId")

        newId_lut = -np.ones(items.movieId.max() + 2, dtype=int)
        newId_lut[items.movieId.values] = items.index.values

        popularity = np.asarray(ratings.sum(axis=0)).flatten()
        index = np.arange(len(popularity), dtype=int)
        items.loc[index, "popularity"] = popularity
        items["normalized_popularity"] = (
            items["popularity"]/(items["year"].max() + 1 - items["year"]))
        items["popularity_rank"] = items["popularity"].rank(method="first", ascending=False)
        items["popularity_log"] = np.log(items["popularity"])
        self.ratings = ratings
        self.newIdLookupTable = newId_lut
        self._tags = "NOT_INITIALIZED"
        self.items = items
        self.ratings_file = f_ratings
        self.f_tags = f_tags
        self._version = version
        self.n_items = ratings.shape[1]
        self.n_users = ratings.shape[0]

    def ml_id_to_internal_id(self, ml_ids):
        return np.take(self.newIdLookupTable, ml_ids, mode='clip')

    def internal_id_to_ml_id(self, internal_ids):
        return self.items.loc[internal_ids]["movieId"]

    def load_rankings(self, file, a_col, b_col, label_col, verbose=False):
        rankings = pd.read_csv(file)

        rankings["a_iid"] = self.ml_id_to_internal_id(rankings[a_col].values)
        rankings["b_iid"] = self.ml_id_to_internal_id(rankings[b_col].values)

        valid_rankings = rankings[(rankings.a_iid > -1) & (rankings.b_iid > -1)]

        info = collections.OrderedDict()
        info["# rankings"] = rankings[a_col].nunique()
        info["# invalid rankings"] = (rankings.groupby(by=a_col).max()["a_iid"] == -1).sum()
        info["# invalid results"] = (rankings["b_iid"] == -1).sum()
        per_ranking = valid_rankings.groupby(by="a_iid").count()["b_iid"]
        labels_per_ranking = valid_rankings.groupby(by="a_iid").sum()[label_col]
        info["Length of each ranking"] = ("%.2f (min=%d, max=%d))" % (
            per_ranking.mean(), per_ranking.min(), per_ranking.max()))
        info["Pos labels in each ranking"] = ("%.2f (min=%d, max=%d))" % (
            labels_per_ranking.mean(), labels_per_ranking.min(), labels_per_ranking.max()))
        if verbose:
            print(info)

        df = pd.DataFrame({"a_iid": valid_rankings["a_iid"], "a_ml_id": valid_rankings[a_col],
                           "b_iid": valid_rankings["b_iid"], "b_ml_id": valid_rankings[b_col],
                           "label": valid_rankings[label_col]})

        As, Bs = list(), list()
        for name, group in df.groupby("a_iid"):
            labels = np.zeros(self.n_items)
            labels[group["b_iid"].values.flatten()] = group["label"].values.flatten()
            As.append(name)
            Bs.append(labels)
        groups = pd.DataFrame({"a_iid": As, "labels": Bs})

        return {"rows": df, "groups": groups}

    def settings(self):
        return {"ml_version": self._version.replace("ml-", "")}

    def get_title(self, internal_id):
        return self.items.loc[internal_id, "title"]

    def get_popularity(self, internal_id):
        return self.items.loc[internal_id, "popularity"]

    def tags(self):
        # lazy loading
        if self._tags == "NOT_INITIALIZED":
            if self.f_tags is not None:
                self._tags = self.read_genome(self.f_tags)
            else:
                self._tags = None

        return self._tags

    def read_genome(self, f_tags):
        ratings_raw = pd.read_csv(f_tags)
        i = ratings_raw["movieId"].values
        j = ratings_raw["tagId"].values
        data = (ratings_raw["relevance"]).astype(np.float32)

        i = self.ml_id_to_internal_id(i)
        valid = (i != -1)
        i = i[valid]
        j = j[valid]
        data = data[valid]

        tags = np.asarray(sp.coo_matrix((data, (i, j))).todense())
        if tags.shape[0] != self.n_items:
            print("Warning: padding Tag Genome file")
            zeros = np.zeros((self.n_items - tags.shape[0], tags.shape[1]))
            tags = np.vstack([tags, zeros])
        return tags


def read_movie_titles(file):
    if "100k" in file:
        df = pd.read_csv(file, sep="|", engine="python", usecols=[
                         0, 1], header=None, names=["movieId", "title"])
    elif "1m" in file or "10m" in file:
        df = pd.read_csv(file, sep="::", engine="python", usecols=[
                         0, 1, 2], header=None, names=["movieId", "title", "genres"])
    else:
        df = pd.read_csv(file)
    return df


def extract_timestamp_features(timestamps, remaining_movie_ids):
    print("Converting timestamps...")
    timestamps = timestamps.T.tolil()
    t_index = pd.date_range(start="1996", end="2020", freq='A-DEC')

    results = list()
    for i, mid in enumerate(remaining_movie_ids):
        df = pd.DataFrame({"time": pd.to_datetime(
            timestamps.data[mid], unit="s")}).set_index('time')
        df["event"] = 1
        ts = df.resample('Y').count().reindex(t_index).fillna(0)
        skew = ts.event.skew()
        peak = ts.event.idxmax().year
        coeffvar = ts["event"].std()/ts["event"].mean()
        first = ts.event.ne(0).idxmax().year
        results += [{"movieId": mid, "time_peak": peak,
                     "time_first": first, "time_skew": skew, "tm_var": coeffvar}]

        if (i % 50) == 0:
            print("Progress:  %d / %d" % (i, len(remaining_movie_ids)))

    return pd.DataFrame(results).set_index("movieId")


def convert_to_implicit(version, rating_threshold=3.0, min_freq=30):
    f_ratings, f_titles, f_tags = ml(version)
    ratings, timestamps = read_movielens_ratings(f_ratings)
    titles = read_movie_titles(f_titles)

    print("Loaded %d entries from %d users with %d movies" %
          (ratings.nnz, ratings.shape[0], ratings.shape[1]))
    print("Movie title database: %d entries" % len(titles))
    ratings.data = (ratings.data >= rating_threshold).astype(int)
    popularity = (ratings.sum(0) >= min_freq)  # movie popularity
    ratings = ratings.multiply(popularity)
    timestamps = timestamps.multiply(ratings)

    timestamps.eliminate_zeros()
    ratings.eliminate_zeros()

    remaining_movie_ids = np.unique(ratings.col)
    new_ids = -np.ones(ratings.shape[1], dtype=int)
    new_ids[remaining_movie_ids] = np.arange(remaining_movie_ids.size)

    ratings.col = new_ids[ratings.col]
    print("After thresholding: %d entries, %d movies remain" %
          (ratings.nnz, len(remaining_movie_ids)))
    path = "/".join(f_ratings.split("/")[:-1]) + "-implicit/"
    Path(path).mkdir(parents=True, exist_ok=True)

    ratings = ratings.tocoo()
    all = np.hstack((ratings.row.reshape(-1, 1), ratings.col.reshape(-1, 1),
                     ratings.data.reshape(-1, 1)))
    np.savez_compressed(path + "ratings.npz", ratings=all)
    np.savetxt(path + "ratings.csv", all, fmt="%d", delimiter=",")

    titles["newId"] = new_ids[titles.movieId]
    titles["year"] = titles["title"].str.extract(r'\((\d{4})\)').fillna("-1").astype(int)
    timestamp_features = extract_timestamp_features(timestamps, remaining_movie_ids)
    titles = titles.join(timestamp_features, how="left", on="movieId")
    titles.to_csv(path + "movies.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("movielens_version", type=str, choices=["100k", "20m", "25m"],
                        help="increase output verbosity")
    parser.add_argument("--min_rating", type=float, default=3.0,
                        help="ratings >= min_rating will be considered positive")
    parser.add_argument("--min_count", type=int, default=30,
                        help="movies with < min_count positive ratings will be removed")
    args = parser.parse_args()
    convert_to_implicit(args.movielens_version, args.min_rating, args.min_count)
