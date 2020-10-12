import lib
from lib.recommenders import *

METHODS = [SklearnRec, SVD, Random, ItemKNN, Cooccur, Popularity, ALS, BPR, SLIM]
OUTFILE = "results/2a_find_best.json"

dataset = lib.ImplicitMovieLens("ml-25m-implicit")
metric = lib.eval.RecallAtK(100)

supervised_trainset = dataset.load_rankings(
    "datasets/labeled/similarity_judgements.train.csv", "movieId", "neighborId", "sim_bin", verbose=True)
supervised_testset = dataset.load_rankings(
    "datasets/labeled/similarity_judgements.dev.csv", "movieId", "neighborId", "sim_bin", verbose=True)

results = dict()
for model in METHODS:
    print("\n\nCurrent model " + model.__name__)
    result = model.grid_search(dataset, supervised_trainset,
                               supervised_testset, metric, verbose=True)

    results[model.__name__] = result.to_dict()
    lib.to_json(OUTFILE, results)
