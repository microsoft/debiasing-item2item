import lib
from lib.recommenders import *
import collections
import pandas as pd

# Methods to demo
METHODS = [lambda x, y: DebiasedModel(x, y), lambda x, y: ItemKNN(x, y, alpha=0.3, lmbda=0.0)]

PRINT_TOP_K = 10


def load_models(unlabeled_data, labeled_data):
    models = collections.OrderedDict()
    for model_constructor in METHODS:
        model = model_constructor(unlabeled_data, labeled_data)
        model.fit()
        models[model.__class__.__name__] = model
    return models


def print_candidates(models, movie, k=PRINT_TOP_K):
    results = collections.OrderedDict()
    for name, model in models.items():
        results[name] = model.get_scored_candidates(movie, k=k).to_dict('records')

    # format side-by-side
    dct = collections.OrderedDict()
    for method, v in results.items():
        for col in ["title", "score"]:
            dct[(method, col)] = pd.DataFrame(v)[col].values
    print(pd.DataFrame(dct).to_string())


def search(dataset, movie_title):
    df = dataset.items[dataset.items.index > -1]
    matches = df[df["title"].str.contains(movie_title, na=False, case=False, regex=False)
                 ].sort_values("popularity", ascending=False)
    matches = matches["title"].iloc[:10].copy().reset_index(drop=False)
    if len(matches):
        matches.index.name = "option #"
        return matches
    else:
        return None


def get_valid_int(max):
    valid = [str(x) for x in range(max+1)]
    while True:
        res = input("Input option (0-%d) [empty to exit]: " % max)
        if not res.strip():
            return None
        elif res in valid:
            return int(res)
        else:
            print("Invalid option. Try again.")


movielens = lib.ImplicitMovieLens("ml-25m-implicit")
similarity_judgements = movielens.load_rankings(
    "datasets/labeled/similarity_judgements.train.csv", "movieId", "neighborId", "sim_bin", verbose=False)

models = load_models(movielens, similarity_judgements)

while True:
    title = input("Input (partial) movie title [empty to quit]: ")
    if not title.strip():
        break

    results = search(movielens, title)
    if results is None:
        print("No matching movies found. Please try again.")
        continue
    else:
        print(results["title"].to_string() + "\n")
        option = get_valid_int(len(results))
        if option is None:
            break

        print("Recommendations for ", results.loc[option]["title"])
        print_candidates(models, results.loc[option]["newId"])
