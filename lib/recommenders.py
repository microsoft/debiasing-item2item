import pandas as pd
import numpy as np
import lib
import implicit.bpr
import sklearn.metrics
import sklearn.ensemble
import sklearn.decomposition
import json
import time
import itertools
import re
import sklearn.linear_model
import os
from . import prop_ranker as prop
import collections

MODEL_CACHE = "models/"


class Cache(object):
    def __init__(self, instance):
        class_name = instance.__class__.__name__
        self._instance = instance
        self.base_path = MODEL_CACHE + "/" + class_name + "/"
        self.model_index = self.base_path + "available_models.json"
        self._loaded_file = None
        if not os.path.exists(self.base_path):
            os.makedirs(self.base_path)
        if os.path.isfile(self.model_index):
            with open(self.model_index, encoding='utf-8') as fh:
                self._models = json.load(fh)
        else:
            self._models = dict()

    def load(self):
        id = self._instance.get_id()
        if id in self._models:
            modelfile = self._models[id]
            modelfile = self.base_path + modelfile
            if os.path.isfile(modelfile):
                self._loaded_file = self._models[id]
                return np.load(modelfile)
            else:
                del self._models[id]
                self.write_index()
        else:
            return None

    def store(self):
        id = self._instance.get_id()
        if not self._loaded_file:
            filename = re.sub("[^0-9a-zA-Z=.,]+", "", id) + "_" + str(int(time.time())) + ".npz"
            self._models[id] = filename
        else:
            filename = self._loaded_file

        np.savez(self.base_path + filename, **self._instance.get_params(), id=self._instance.get_id())
        self._loaded_file = filename
        self.write_index()

    def write_index(self):
        with open(self.model_index, 'w', encoding='utf8') as json_file:
            json.dump(self._models, json_file, ensure_ascii=False)

    def rebuild_index(self):
        new_index = dict()
        for file in os.listdir(self.base_path):
            if file.endswith(".npz"):
                id = np.load(self.base_path + file)["id"].tolist()
                if id in new_index:
                    print("Warning - Following id already exists: ", id)
                new_index[id] = file
        self._models = new_index
        self.write_index()


class ScoreTable(object):
    def __init__(self, results):
        self._df = pd.DataFrame(results)
        self._argmax = self._df.loc[self._df["score"].idxmax()]

    def best_param(self):
        return self._argmax["params"]

    def best_score(self):
        return self._argmax["score"]

    def __str__(self):
        return self._df.to_string()

    def to_dict(self):
        return {"best_param": self.best_param(), "best_score": self.best_score(), "all_scores": self._df.to_dict('records')}


class Recommender(object):
    PARAM_GRID = dict()

    @classmethod
    def grid_search(cls, implicit, supervised_train, supervised_val, metric, verbose=False):
        def product_dict(kwargs):
            keys = kwargs.keys()
            vals = kwargs.values()
            for instance in itertools.product(*vals):
                yield dict(zip(keys, instance))

        results = list()
        for param_id, params in enumerate(product_dict(cls.PARAM_GRID)):
            if verbose:
                print("Fitting ", params)
            model = cls(implicit, supervised_train, **params)
            model.fit()
            val_score = list(model.evaluate(supervised_val, [metric]).values())[0]["mean"]
            results += [{"id": param_id, "params": params, "score": val_score}]

        table = ScoreTable(results)
        if verbose:
            print(table)
        return table

    def __init__(self, data, supervised_trainset=None):
        self._model = None
        self._my_settings = dict()
        self._data = data

    def get_id(self):
        return json.dumps(self.settings(), ensure_ascii=False, sort_keys=True)

    def print_similar_items(self, item_id, k=10):
        self.print_items(item_id, self.similar_items(item_id, k))

    def print_scored_candidates(self, item_id, candidate_ids=None, k=10):
        candidate_ids, scores = self.score_candidates(item_id, candidate_ids, rank=True)
        self.print_items(item_id, candidate_ids[:k], scores[:k])

    def get_scored_candidates(self, item_id, candidate_ids=None, k=10):
        candidate_ids, scores = self.score_candidates(item_id, candidate_ids, rank=True)
        return self.items_to_df(item_id, candidate_ids[:k], scores[:k])

    def settings(self):
        return {"data": self._data.settings(), **self._settings()}

    def _settings(self):
        return dict(self._my_settings)

    def fit(self):
        self._fit()

    def print_items(self, movie_id, candidate_ids, scores):
        print("Scored candidates for %s" % self._data.items["title"][movie_id])
        print(self.items_to_df(movie_id, candidate_ids, scores))

    def items_to_df(self, movie_id, candidate_ids, scores):
        titles = self._data.items["title"]
        popranks = self._data.items["popularity_rank"]
        return pd.DataFrame({"title": [titles[m] for m in candidate_ids], "score": scores, "movieId": candidate_ids, "poprank": [popranks[m] for m in candidate_ids]})

    def score_candidates(self, item_id, candidate_ids=None, rank=False):
        if candidate_ids:
            candidate_ids = np.asarray(candidate_ids)
        else:
            candidate_ids = np.arange(self._data.n_items)
        scores = self._score_candidates(item_id, candidate_ids)
        scores[candidate_ids == item_id] = -np.inf  # exclude item itself
        if rank:
            argsort = np.argsort(-scores)
            scores_sorted = scores[argsort]
            candidates_sorted = candidate_ids[argsort]
            return candidates_sorted, scores_sorted
        else:
            return candidate_ids, scores

    def evaluate(self, rankings, metrics):
        rankings = rankings["groups"]
        if not len(rankings):
            raise ValueError("Length of rankings must be at least one!")
        scores = collections.defaultdict(list)
        for _, ranking in rankings.iterrows():
            candidate_ids, pred_score = self.score_candidates(ranking["a_iid"])
            for metric in metrics:
                scores[metric.name()] += [metric.score(ranking["labels"], pred_score)]
        return {k: {"mean": np.mean(v), "raw": v} for k, v in scores.items()}

    def create_debug_file(self, rankings, k=10, features=["popularity", "year", "time_first"]):
        def create_dict(m_iid, score, relevance, rank, ab=-1):
            _features = {f: self._data.items.loc[m_iid, f] for f in features}
            return {"rank": rank, "title": self._data.items.loc[m_iid, "title"],
                    "relevance": relevance, "score": score, "counts(a|b)": ab,
                    **_features}

        rankings = rankings["groups"]
        reports = list()
        cc = Cooccur(self._data)
        cc.fit()

        for _, ranking in rankings.iterrows():
            report = list()
            candidate_ids, pred_score = self.score_candidates(ranking["a_iid"])
            df = pd.DataFrame({"m_iid": candidate_ids, "score": pred_score,
                               "label": ranking["labels"].tolist()})
            df['rank'] = df["score"].rank(method="first", ascending=False)
            df['c_ab'] = cc.counts_ab(ranking["a_iid"], np.arange(self._data.n_items))
            df.sort_values(by='rank', inplace=True)
            report += [create_dict(ranking["a_iid"], np.NaN, -1, -1)]

            report += [{}]
            for i, row in df.iloc[:k].iterrows():
                report += [create_dict(int(row["m_iid"]), row["score"],
                                       row["label"], row["rank"], row["c_ab"])]

            report += [{}]
            for i, row in df[df["label"] > 0].iterrows():
                report += [create_dict(int(row["m_iid"]), row["score"],
                                       row["label"], row["rank"], row["c_ab"])]
            report = pd.DataFrame(report)
            for c in ["rank", "relevance", "year", "time_first", "popularity", "counts(a|b)"]:
                report[c] = report[c].astype("Int64")

            reports += [report.to_string(
                formatters={"title": lambda x: (str(x) + " "*50)[:35]}, index=False)]
        with open(self.__class__.__name__ + ".debug.txt", 'w', encoding='utf8') as file:
            file.write("\n\n".join(reports))


class CachedRecommender(Recommender):
    def __init__(self, data, supervised_trainset=None):
        super().__init__(data)
        self._cache = Cache(self)

    def fit(self):
        cached_params = self._cache.load()
        if cached_params is None:
            print("Couldn't find saved model. Training.")
            self._fit()
            self._cache.store()
        else:
            print("Found cached model.")
            cached_params = {k: v for k, v in cached_params.items() if k != "id"}
            self.set_params(cached_params)


class Cooccur(Recommender):
    def __init__(self, data, supervised_trainset=None):
        super().__init__(data)
        self._ratings = data.ratings
        self._data = data

    def _fit(self):
        marginal = np.asarray(self._ratings.sum(axis=0))
        marginal[marginal == 0] = 1  # prevent division by 0
        self._norms = marginal.flatten()
        self._item_factors = self._ratings.T

    def _score_candidates(self, item_id, candidate_ids):
        return self.counts_ab(item_id, candidate_ids)/self.counts_b(item_id)

    def counts_ab(self, item_id, candidate_ids):
        item = self._item_factors[item_id]
        candidates = self._item_factors[candidate_ids]
        return np.asarray(candidates.dot(item.T).todense()).flatten()

    def counts_b(self, item_id):
        return self._norms[item_id]


class ItemKNN(Cooccur):
    PARAM_GRID = {'alpha': [0.3, 0.5, 0.7], 'lmbda': [0.0, 10.0, 20.0]}

    def __init__(self, data, supervised_data=None, alpha=0.5, lmbda=0.0):
        super().__init__(data)
        self._my_settings["alpha"] = alpha
        self._my_settings["lmbda"] = lmbda

    def _fit(self):
        self._norms = np.asarray(self._ratings.sum(axis=0)).flatten()
        self._item_factors = self._ratings.T

    def _score_candidates(self, item_id, candidate_ids):
        alpha, lmbda = self._my_settings["alpha"], self._my_settings["lmbda"]
        item = self._item_factors[item_id]
        candidates = self._item_factors[candidate_ids]
        joint = np.asarray(candidates.dot(item.T).todense())

        candidate_norms = np.power(self._norms[candidate_ids] + lmbda, 1.0 - alpha)
        norms = candidate_norms

        item_norm = np.power(self._norms[item_id] + lmbda, alpha)
        norms *= item_norm

        norms[norms == 0] = 1

        return joint.flatten()/norms


class Random(Recommender):
    def __init__(self, data, supervised_trainset=None):
        self._data = data

    def _fit(self):
        pass

    def _score_candidates(self, movie_id, candidate_ids):
        rng = np.random.RandomState(movie_id)
        return rng.rand(len(candidate_ids))


class Popularity(Recommender):
    def _fit(self):
        pass

    def _score_candidates(self, movie_id, candidate_ids):
        return self._data.get_popularity(candidate_ids).values


class SLIM(CachedRecommender):
    PARAM_GRID = {'l1_reg': [0.001, 0.01], 'l2_reg': [0.001, 0.01]}

    def __init__(self, data, supervised_data=None, l1_reg=0.001, l2_reg=0.01):
        super().__init__(data)
        self._my_settings["l1_reg"] = l1_reg
        self._my_settings["l2_reg"] = l2_reg

        self._ratings = data.ratings.T.tolil()
        self._data = data
        self._item_weights = dict()

    def _fit(self):
        l1_reg = self._my_settings["l1_reg"]
        l2_reg = self._my_settings["l2_reg"]
        alpha = l1_reg+l2_reg
        l1_ratio = l1_reg/alpha

        self._model = sklearn.linear_model.ElasticNet(
            alpha=alpha, l1_ratio=l1_ratio, positive=True, fit_intercept=False)

    def get_params(self):
        return {str(k): v for k, v in self._item_weights.items()}

    def set_params(self, cached_params):
        self._item_weights = {int(k): v for k, v in cached_params.items()}

    def _score_candidates(self, item_id, candidate_ids):
        if item_id not in self._item_weights:
            self._fit()
            data = self._ratings[item_id]
            self._ratings[item_id] = 0
            self._model.fit(self._ratings.T, data.toarray().reshape(-1))
            self._ratings[item_id] = data
            self._item_weights[item_id] = self._model.coef_.copy()
            self._cache.store()

        return self._item_weights[item_id][candidate_ids]


class BPR(CachedRecommender):
    PARAM_GRID = {'n_factors': [50, 100, 150], 'lmbda': [0.01, 0.1]}

    def __init__(self, data, supervised_data=None, n_factors=10, lmbda=0.01):
        super().__init__(data)

        self._ratings = data.ratings.T

        self._my_settings["n_factors"] = n_factors
        self._my_settings["lambda"] = lmbda

    def get_params(self):
        return {"item_factors": self._item_factors, "norms": self._norms}

    def set_params(self, cached_params):
        self._item_factors = cached_params["item_factors"]
        self._norms = cached_params["norms"]

    def _fit(self):
        self._model = implicit.bpr.BayesianPersonalizedRanking(
            regularization=self._my_settings["lambda"], factors=self._my_settings["n_factors"])
        self._model.fit(self._ratings)
        self._item_factors = self._model.item_factors
        self._fit_norms()

    def _fit_norms(self):
        norms = np.linalg.norm(self._item_factors, keepdims=True, axis=1)
        norms[norms == 0] = 10**-10  # to prevent Inf in cosine similarity
        self._norms = norms.squeeze()

    def similar_items(self, movie_id, k=10):
        return self._model.similar_items(movie_id, k)

    def _score_candidates(self, item_id, candidate_ids):
        item = self._item_factors[item_id]
        candidates = self._item_factors[candidate_ids]
        return candidates.dot(item)/self._norms[candidate_ids]


class ALS(BPR):
    def _fit(self):
        self._model = implicit.als.AlternatingLeastSquares(
            regularization=self._my_settings["lambda"], factors=self._my_settings["n_factors"])
        self._model.fit(self._ratings)
        self._item_factors = self._model.item_factors
        self._fit_norms()


class SVD(BPR):
    PARAM_GRID = {'n_factors': [25, 50, 100, 150], 'center_data': [False]}

    def __init__(self, data, supervised_data=None, n_factors=10, center_data=False):
        super().__init__(data, n_factors=n_factors)
        self._my_settings["center_data"] = center_data
        del self._my_settings["lambda"]

    def _fit(self):
        if self._my_settings["center_data"]:
            self._model = sklearn.decomposition.PCA(
                n_components=self._my_settings["n_factors"], copy=False)
            self._item_factors = self._model.fit_transform(self._ratings.todense())
        else:
            self._model = sklearn.decomposition.TruncatedSVD(
                n_components=self._my_settings["n_factors"], random_state=42)
            self._item_factors = self._model.fit_transform(self._ratings)

        self._fit_norms()


class CosineNN(Recommender):
    def similar_items(self, item_id, k=10):
        item = self._item_factors[item_id]
        scores = self._item_factors.dot(item).squeeze()
        argsort = np.argsort(-scores)
        sort = np.take_along_axis(scores, argsort, axis=-1)
        return zip(argsort[:k], sort[:k])

    def _score_candidates(self, item_id, candidate_ids):
        item = self._item_factors[item_id]
        candidates = self._item_factors[candidate_ids]
        return candidates.dot(item)


class TagGenome(CosineNN):
    def _fit(self):
        self._item_factors = self._data.tags()
        norms = np.linalg.norm(self._item_factors, keepdims=True, axis=1)
        norms[norms == 0] = 10**-10  # to prevent Inf in cosine similarity
        self._item_factors /= norms


class Genre(CosineNN):
    def _fit(self):
        items = self._data.items
        self._item_factors = items.loc[np.arange(items.index.max()+1, dtype=int),
                                       "genres"].str.get_dummies(sep="|").to_numpy()
        norms = np.linalg.norm(self._item_factors, keepdims=True, axis=1)
        norms[norms == 0] = 10**-10  # to prevent Inf in cosine similarity
        self._norms = norms
        self._item_factors = self._item_factors.astype(float)/norms


class DebiasedModel(Recommender):
    def __init__(self, data, labeled_rankings, model="ranker"):
        self._data = data
        self._cc = Cooccur(self._data)
        self._cc.fit()
        self._model = model
        self._labeled_rankings = labeled_rankings
        self._raw_features = ["popularity_log", "year", "time_first"]

    def _prepare_for_propensity_learning(self, rankings):
        cc = self._cc

        def augment(grp):
            b = grp.name
            a = grp["b_iid"].values
            grp["counts_ab"] = cc.counts_ab(b, a)
            grp["counts_b"] = cc.counts_b(b)

            grp["features"] = list(self._get_features(b, a))
            return grp

        rankings = rankings.groupby("a_iid").apply(augment)
        rankings["MLE_est"] = rankings["counts_ab"]/rankings["counts_b"]
        return rankings

    def _add_negative_samples(self, rankings, n_neg=500):
        def add_negative(grp):
            pos = grp[grp["label"] > 0].reset_index(drop=True)
            positive_ids = np.append(pos["b_iid"].values, [grp.name])

            # sample negative items at random
            negative_ids = np.setdiff1d(np.arange(self._data.n_items), positive_ids)
            neg_samples = np.random.choice(negative_ids, n_neg, replace=False)

            neg_df = pd.DataFrame({"a_iid": [grp.name]*len(neg_samples),
                                   "b_iid": neg_samples, "label": np.zeros_like(neg_samples)})
            df = pd.concat([pos, neg_df])
            return df

        return rankings.groupby("a_iid").apply(add_negative).reset_index(drop=True)

    def _learn_propensities(self):
        labeled_rankings = self._labeled_rankings["rows"]
        labeled_rankings = self._add_negative_samples(labeled_rankings)

        rankings = self._prepare_for_propensity_learning(labeled_rankings)
        cols = {"offset": "MLE_est", "features": "features", "y": "label"}

        X, y, offset = prop.RankingToPairs(column_mapping=cols).transform(rankings)
        print("Transformed to %d pairs" % len(y))
        ranker = prop.Ranker(model="linear", lr=0.02, n_steps=300)

        ranker.fit(X, y, offset)
        ranker.print_params()
        self._ranker = ranker

        mle = rankings["MLE_est"].values.flatten()
        features = np.asarray(rankings[cols["features"]].tolist())
        rankings["p_inv_pred"] = ranker.transform(features)
        rankings["MLE_IPS"] = ranker.rank(features, mle)

        return rankings

    def _fit(self):
        rankings = self._learn_propensities()
        self.debug(rankings)

    def debug(self, rankings, examples_to_show=0):
        for name, grp in itertools.islice(rankings.groupby("a_iid"), examples_to_show):
            print(self._data.get_title(name))

            my_grp = grp.copy()
            my_grp["b_name"] = self._data.get_title(grp["b_iid"].values).values
            my_grp["MLE_IPS_rank"] = my_grp["MLE_IPS"].rank(ascending=False)
            print("debug recall", lib.eval.RecallAtK(50).score(
                my_grp["label"].values, my_grp["MLE_IPS"].values))

            print(my_grp[["b_name", "label", "MLE_IPS_rank", "MLE_est", "p_inv_pred"] +
                         ["counts_ab", "counts_b"]].sort_values("label", ascending=False).iloc[:20].to_string())

        recall_vals = list()
        for name, grp in rankings.groupby("a_iid"):
            recall_vals += [lib.eval.RecallAtK(50).score(grp["label"].values,
                                                         grp["MLE_IPS"].values)]
        print("After training, have recall of", np.mean(recall_vals))

    def _get_features(self, item_id, candidate_ids):
        features = list()
        data = self._data
        for att in self._raw_features:
            features += [data.items.loc[candidate_ids, att].values]
            features += [abs(data.items.loc[candidate_ids, att].values -
                             data.items.loc[item_id, att])]

        cc = self._cc
        features += [cc.counts_ab(item_id, candidate_ids)/cc.counts_b(candidate_ids)]
        features = np.asarray(features).T
        return features

    def _score_candidates(self, item_id, candidate_ids):
        mle = self._cc._score_candidates(item_id, candidate_ids)
        X = self._get_features(item_id, candidate_ids)
        return self._ranker.rank(X, mle)


class SklearnRec(DebiasedModel):
    PARAM_GRID = {'n_estimators': [50, 100], 'max_depth': [2, 3, 5]}

    def __init__(self, data, labeled_rankings, method="random", n_estimators=100, max_depth=3):
        self._data = data
        self._cc = Cooccur(self._data)
        self._cc.fit()
        self._method = method.lower()
        self._labeled_rankings = labeled_rankings
        self._my_settings = {"n_estimators": n_estimators, "max_depth": max_depth}
        self._raw_features = ["popularity_log", "year", "time_first"]

    def _fit(self):
        labeled_rankings = self._labeled_rankings["rows"]
        labeled_rankings = self._add_negative_samples(labeled_rankings)

        rankings = self._prepare_for_propensity_learning(labeled_rankings)

        ranker = sklearn.ensemble.GradientBoostingClassifier(random_state=0, **self._my_settings)
        features = np.asarray(rankings["features"].tolist())
        X = np.hstack((features, rankings["MLE_est"].values.reshape(-1, 1)))
        y = labeled_rankings["label"].values

        ranker.fit(X, y)
        self._ranker = ranker

        return rankings

    def _score_candidates(self, item_id, candidate_ids):
        mle = self._cc._score_candidates(item_id, candidate_ids)
        features = self._get_features(item_id, candidate_ids)
        X = np.hstack((features, mle.reshape(-1, 1)))
        return self._ranker.predict_proba(X)[:, 1].flatten()
