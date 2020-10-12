import numpy as np
import sklearn.metrics


class Metric(object):
    pass


class BinaryMetric(Metric):
    def __init__(self, k=-1):
        self._k = k

    def _parse_vars(self, y_true, y_score):
        if not isinstance(y_true, (list, np.ndarray)) or not isinstance(y_score, (list, np.ndarray)):
            raise ValueError("Input arrays need to be either list or numpy array")

        y_true = np.asarray(y_true, dtype=int)
        y_score = np.asarray(y_score)

        if y_true.ndim != 1 or y_score.ndim != 1:
            raise ValueError("Input arrays need to be 1D.")
        if y_true.size != y_score.size:
            raise ValueError("Both arrays need to be same size.")
        if y_true.max() > 1 or y_true.min() < 0:
            raise ValueError("y_true may only contain {0, 1}.")
        if not np.all(np.isfinite(y_score)):
            y_score[~np.isfinite(y_score)] = -np.inf
        return y_true, y_score

    def name(self):
        return self.__class__.__name__.replace("K", str(self._k))

    def score(self, y_true, y_score):
        y_true, y_score = self._parse_vars(y_true, y_score)
        k = self._k if self._k > 0 else len(y_true)

        n_pos = np.sum(y_true == 1)
        order = np.argsort(y_score)[::-1]
        y_pred_labels = np.take(y_true, order[:k])
        n_relevant = np.sum(y_pred_labels == 1)
        return self._score(n_pos, n_relevant, k, y_pred_labels)


class RecallAtK(BinaryMetric):
    def _score(self, n_pos, n_relevant, k, y_pred_labels):
        if n_pos == 0:
            return 0.0  # if there are no relevant items, return 0
        else:
            return float(n_relevant) / n_pos


class SumOfRanks(BinaryMetric):
    def _score(self, n_pos, n_relevant, k, y_pred_labels):
        if n_pos == 0:
            return 0.0  # if there are no relevant items, return 0
        else:
            return (1.0 + np.nonzero(y_pred_labels)[0]).mean()


class PrecisionAtK(BinaryMetric):
    def _score(self, n_pos, n_relevant, k, y_pred_labels):
        return float(n_relevant) / k


class DCG(Metric):
    def score(self, y_true, y_pred):
        return sklearn.metrics.dcg_score([y_true], [y_pred])


class NDCG(Metric):
    def score(self, y_true, y_pred):
        return sklearn.metrics.ndcg_score([y_true], [y_pred])


class KendallTau(Metric):
    def score(self, y_true, y_pred):
        def sign(x):
            if x >= 0:
                return 1
            else:
                return -1

        numer = 0
        denom = 0
        n = len(y_true)
        for i in range(2, n):
            for j in range(1, i - 1):
                if y_true[i] != y_true[j]:  # no tie
                    numer = numer + sign(y_true[i]-y_true[j])*sign(y_pred[i] - y_pred[j])
                    denom += 1
        return numer/float(denom)
