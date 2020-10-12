import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import sklearn.preprocessing
import sklearn.ensemble
import sklearn.linear_model
import sklearn.model_selection
import sklearn.metrics
import sklearn.pipeline

import sklearn.neighbors
import torch.nn.utils.rnn

np.random.seed(42)
torch.manual_seed(42)


def no_warn_log(x):
    result = -np.ones_like(x)*np.inf
    result[x > 0] = np.log(x[x > 0])
    return result


class Ranker(object):
    models = {"linear": lambda dim: torch.nn.Linear(dim, 1, bias=False),
              "2layer": lambda dim: torch.nn.Sequential(
        torch.nn.Linear(dim, 50),
        torch.nn.LeakyReLU(),
        torch.nn.Linear(50, 1),
    ),
        "3layer": lambda dim: torch.nn.Sequential(
        torch.nn.Linear(dim, 40),
        torch.nn.LeakyReLU(),
        torch.nn.Linear(40, 5),
        torch.nn.LeakyReLU(),
        torch.nn.Linear(5, 1),
    )
    }

    def __init__(self, lr=0.5, n_steps=1000, verbose=True, model="linear"):
        self._lr = lr
        self._n_steps = n_steps
        self._params = None
        self._verbose = verbose
        poly = sklearn.pipeline.make_pipeline(sklearn.preprocessing.StandardScaler(
        ), sklearn.preprocessing.PolynomialFeatures(2), sklearn.preprocessing.StandardScaler())
        self._scaler = sklearn.pipeline.FeatureUnion([("poly", poly)])
        self._scaler = sklearn.preprocessing.StandardScaler()
        self._model = self.models[model]

    def fit(self, X, y, offset):
        X_i = torch.tensor(self._scaler.fit_transform(X[0]), dtype=torch.double)
        X_j = torch.tensor(self._scaler.transform(X[1]))

        offset = torch.tensor(offset.reshape(-1, 1), dtype=torch.double)
        model = self._model(X_i.size(1)).double()

        if self._verbose:
            print("Length of input", X_i.shape)

        def pred():
            output = model(X_i) - model(X_j)
            return output + offset

        def loss():
            loss = F.relu(-pred())
            loss = loss.mean()
            _optimizer.zero_grad()
            loss.backward()
            return loss

        def percentage_violated():
            return (pred() < 0).float().sum().item()

        _optimizer = torch.optim.Adam(model.parameters(), lr=self._lr)
        if self._verbose:
            all = pred()
            invalid = all[~torch.isfinite(all)]
            if len(invalid):
                print("Warning: Found ", len(invalid), "invalid pairs")
            print("\tloss\t#constraints violated")
        for step in range(self._n_steps):
            current_loss = _optimizer.step(loss)
            if (step % 50 == 0 or step == self._n_steps - 1) and self._verbose:
                print("", "%.2e" % current_loss.item(), "%.2f" % percentage_violated(), sep="\t")
        self._model = model

        self._params = {n: p.data.cpu().numpy().T for n, p in model.named_parameters()}

    def print_params(self):
        print(self._params)

    def rank(self, X, mle):
        p_inv = self.transform(X, exp=False).flatten()
        return no_warn_log(mle.flatten()) + p_inv

    def transform(self, X, exp=True):
        X = self._scaler.transform(X)
        scores = self._model(torch.as_tensor(X)).data.cpu().numpy().flatten()
        if exp:
            return np.exp(scores)
        else:
            return scores


class RankingToGroups(object):
    def transform(self, df):
        pos, neg = list(), list()
        features, mle = None, list()
        bs = np.arange(df["b_iid"].max()+1)
        for name, grp in df.groupby("a_iid"):
            pos += [grp[grp["sim_ab"] > 0]["b_iid"].values]
            neg += [grp[grp["sim_ab"] <= 0]["b_iid"].values]
            mle += [df.set_index("b_iid").loc[bs, "MLE_est"].values]
        features = df.set_index("b_iid").loc[bs, ["att_days", "att_popularity"]].to_numpy()

        return {"pos": pos, "neg": neg, "features": features, "mle": mle}


class RankingToPairs(object):
    def __init__(self, confidence_threshold=None, column_mapping=dict(), binary_labels=True):
        self._threshold = confidence_threshold
        default = {"q": "a_iid", "y": "sim_ab", "counts_b": "counts_b", "offset": "MLE_est",
                                 "counts_ab": "counts_ab", "features": "features"}
        self._column_mapping = {**default, **column_mapping}
        self._binary_labels = binary_labels

    @staticmethod
    def _induce_pairs(indices, vals):
        n = len(vals)
        b, a = np.meshgrid(np.arange(n), np.arange(n))
        valid = (vals[a] > vals[b])
        return indices[a[valid]], indices[b[valid]]

    @staticmethod
    def induce_pairs(y, q):
        df = pd.DataFrame({"q": q, "y": y})
        pairs = list()
        for q_id, grp in df.groupby("q"):
            t1, t2 = RankingToPairs._induce_pairs(grp.index.values, grp.y.values)
            _pairs = list(zip(t1.tolist(), t2.tolist()))
            pairs += _pairs
        pairs = np.asarray(pairs, dtype=int)
        return pairs[:, 0], pairs[:, 1]

    @staticmethod
    def get_conf_interval(n_ab, n_b, N, replications=1000, alpha=0.05):
        n_ab = int(n_ab)
        n_b = int(n_b)
        N = int(N)
        sample = np.random.choice(N, (N, replications))
        v_ab = np.zeros(N)
        v_ab[:n_ab] = 1

        v_b = np.zeros(N)
        v_b[:n_b] = 1

        statistics = v_ab[sample].mean(axis=0)/v_b[sample].mean(axis=0)
        left = np.percentile(statistics, alpha/2*100)
        right = np.percentile(statistics, 100-alpha/2*100)
        if n_ab == 0:
            return (0.0, 1.0)
        else:
            return (left, right)

    def transform(self, df):
        cols = self._column_mapping
        X = np.asarray(df[cols["features"]].tolist())
        q = df[cols["q"]].values
        y = df[cols["y"]].values
        offsets = no_warn_log(df[cols["offset"]].values)
        N = df[cols["counts_b"]].max()*2

        i, j = self.induce_pairs(y, q)
        X_new = (np.take(X, i, axis=0),  np.take(X, j, axis=0))
        y_new = np.ones_like(i)
        offsets_new = np.take(offsets, i, axis=0) - np.take(offsets, j, axis=0)

        filtered = np.isfinite(offsets_new)
        if self._threshold:
            cbs = df.apply(lambda row: pd.Series(self.get_conf_interval(
                row[cols["counts_ab"]], row[cols["counts_b"]], N), index=['lcb', 'ucb']), axis=1)
            cb = cbs["ucb"].values - cbs["lcb"].values
            filtered = (np.take(cb, i, axis=0) <= self._threshold) & (
                np.take(cb, j, axis=0) <= self._threshold)
        return (X_new[0][filtered], X_new[1][filtered]), y_new[filtered], offsets_new[filtered]


def test_ranker(df):
    X, y, offset = RankingToPairs(column_mapping={"q": "a_iid"}).transform(df)
    ranker = Ranker(lr=0.5, n_steps=500, model="linear")
    ranker.fit(X, y, offset)
    return ranker
