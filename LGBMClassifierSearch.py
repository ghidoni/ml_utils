import lightgbm as lgb
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from typing import List, Optional


class LGBMClassifierSearch:
    def __init__(self, cv=None, search_type="grid", search_params=None):
        self.search_type = search_type
        self.clf = lgb.LGBMClassifier(random_state=8, importance_type="gain")

        if search_params is None:
            self.search_params = {
                "learning_rate": [0.01, 0.1],
                "n_estimators": [10, 25, 50, 100],
                "max_depth": [3, 5],
            }
        else:
            self.search_params = search_params

        if cv is None:
            self.folds = KFold(n_splits=5, shuffle=True, random_state=8)
        else:
            self.folds = cv

        # Outputs
        self.best_params = None
        self.train_score = None
        self.test_score = None
        self.gini = None
        self.eval_results = None

    def fit(
        self,
        X,
        y,
        eval_set: Optional[List[tuple]] = None,
        eval_metric: Optional[str] = None,
        callbacks: Optional[List] = None,
    ):
        # n_obs = X.shape[0]

        if self.search_type == "grid":
            self.search = GridSearchCV(
                self.clf,
                self.search_params,
                cv=self.folds.split(X, y),
                scoring="roc_auc",
                n_jobs=-1,
            )
        elif self.search_type == "random":
            self.search = RandomizedSearchCV(
                self.clf,
                self.search_params,
                cv=self.folds.split(X, y),
                scoring="roc_auc",
                n_jobs=-1,
            )
        else:
            raise ValueError("search_type must be 'grid' or 'random'")

        # valid_sets: Optional[List[tuple]] = None
        # if eval_set is not None:
        #     if isinstance(eval_set, tuple):
        #         eval_set = [eval_set]
        #     valid_sets = []
        #     for valid_x, valid_y in eval_set:
        #         valid_sets.append((valid_x, valid_y))

        self.search.fit(
            X,
            y,
            eval_set=eval_set,
            eval_metric=eval_metric,
            callbacks=callbacks,
        )

        # self.train_score = self.search.cv_results_["mean_train_score"]
        # self.test_score = self.search.cv_results_["mean_test_score"]
        # self.best_params = self.search.best_params_
        # y_pred_proba = self.search.predict_proba(X)[:, 1]
        # self.gini = 2 * roc_auc_score(y, y_pred_proba) - 1
        return self

    def predict(self, X):
        return self.search.predict(X)

    def predict_proba(self, X):
        return self.search.predict_proba(X)

    def set_params_clf(self, **params):
        self.clf.set_params(**params)
        return self

    def get_params_clf(self, deep=True):
        return self.clf.get_params(deep)

    def set_params_search(self, **params):
        self.search.set_params(**params)
        return self

    def get_params_search(self, deep=True):
        return self.search.get_params(deep)

    def plot_metric(self, metric="binary_logloss"):
        ax = lgb.plot_metric(self.eval_results, metric=metric, figsize=(10, 8))
        ax.set_title(f"{metric} over iterations")
        plt.show()
