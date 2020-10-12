import lib
from lib.recommenders import *
import pandas as pd

BASELINES = [SklearnRec, SVD, Random, ItemKNN, Cooccur, Popularity, ALS, BPR, SLIM, DebiasedModel]
INFILE = "results/2a_find_best.json"
OUTFILE = "results/2b_eval_on_test.json"

dataset = lib.ImplicitMovieLens("ml-25m-implicit")
metrics = [lib.eval.SumOfRanks(), lib.eval.RecallAtK(
    100), lib.eval.RecallAtK(50), lib.eval.RecallAtK(25)]

supervised_trainset = dataset.load_rankings(
    "datasets/labeled/similarity_judgements.train.csv", "movieId", "neighborId", "sim_bin", verbose=True)
supervised_testset = dataset.load_rankings(
    "datasets/labeled/similarity_judgements.test.csv", "movieId", "neighborId", "sim_bin", verbose=True)
prev_models = lib.from_json(INFILE)

results = dict()
table = list()
for model in BASELINES:
    model_name = model.__name__
    print("\nCurrent model " + model_name)
    if model_name in prev_models:
        best_param = prev_models[model_name]["best_param"]
        print("Best params on val:", best_param)
    else:
        best_param = dict()
        print("No hyperparameters found.")

    best_model = model(dataset, supervised_trainset, **best_param)
    best_model.fit()
    results[model_name] = best_model.evaluate(supervised_testset, metrics)
    results[model_name]["example"] = best_model.get_scored_candidates(0).to_dict('records')
    for metric, scores in results[model_name].items():
        if metric != "example":
            table += [{"model": model_name, "metric": metric, "mean score": scores["mean"]}]

lib.to_json(OUTFILE, results)
print(pd.DataFrame(table).to_string())
