import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV


# What should be metric? maximize validation gini?
# minimize gini difference between train and validation?


def lgb_clf_search(
    X, y, X_val, y_val, cv="default", search_type="grid", search_params=None
):
    n_obs = X.shape[0]

    clf = lgb.LGBMClassifier(random_state=8, importance_type="gain")

    if search_params is None:
        params = {
            "learning_rate": [0.01, 0.1],
            "n_estimators": [10, 25, 50, 100],
            "max_depth": [3, 5],
            "min_child_samples": [(n_obs / 5) * 0.02, (n_obs / 5) * 0.05],
            "reg_lambda": [1, 0.1, 0.01],
        }

    if cv == "default":
        folds = KFold(n_splits=5, shuffle=True, random_state=8)

    if search_type == "grid":
        search = GridSearchCV(
            clf, params, cv=folds.split(X, y), scoring="roc_auc", n_jobs=-1
        )
    elif search_type == "random":
        search = RandomizedSearchCV(
            clf, params, cv=folds.split(X, y), scoring="roc_auc", n_jobs=-1
        )
    else:
        raise ValueError("search_type must be 'grid' or 'random'")

    search.fit(X, y)

    train_score = search.cv_results_["mean_train_score"]
    test_score = search.cv_results_["mean_test_score"]
    best_params = search.best_params_
    y_pred_proba = search.predict_proba(X)[:, 1]
    gini = 2 * roc_auc_score(y, y_pred_proba) - 1

    return train_score, test_score, best_params, gini
