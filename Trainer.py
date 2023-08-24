import optuna
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from joblib import Parallel, delayed


class Trainer:
    def __init__(self, df, features, target):
        self.df = df
        self.features = features
        self.target = target
        self.eval_results = []
        self.eval_df = None

    def custom_metric(self, train, valid):
        alpha = 3
        beta = 0.1
        return alpha * valid - beta * abs(train - valid)

    def objective_optuna(self, trial):
        params = {
            "objective": "binary",
            "metric": "auc",
            "verbosity": -1,
            "boosting_type": "gbdt",
            "max_depth": trial.suggest_int("max_depth", 2, 5, step=1),
            "learning_rate": trial.suggest_float(
                "learning_rate", 0.01, 0.1, log=True
            ),
            "num_leaves": trial.suggest_int("num_leaves", 2, 32, step=2),
            "max_bin": trial.suggest_int("max_bin", 50, 250, step=50),
        }

        model = lgb.LGBMClassifier(
            **params, random_state=8, importance_type="gain"
        )

        opt_callback = optuna.integration.LightGBMPruningCallback(
            trial, "auc", valid_name="valid"
        )

        score_train, score_valid = self.training(model, opt_callback)
        metric = self.custom_metric(score_train, score_valid)
        self.eval_results.append(
            [trial.number, score_train, score_valid, metric]
        )

        return metric

    def training(self, model, callback=None):
        X = self.df[self.features]
        y = self.df[self.target]

        mod = model

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=8)
        mod_callbacks = [
            lgb.early_stopping(10),
            # lgb.log_evaluation(period=100),
        ]

        if callback is not None:
            mod_callbacks.append(callback)

        def train_fold(fold):
            train_idx, valid_idx = fold
            X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
            X_valid, y_valid = X.iloc[valid_idx], y.iloc[valid_idx]

            mod.fit(
                X_train,
                y_train,
                eval_set=[(X_train, y_train), (X_valid, y_valid)],
                eval_names=["train", "valid"],
                eval_metric="auc",
                callbacks=mod_callbacks,
            )

            score_train = mod.best_score_["train"]["auc"]
            score_valid = mod.best_score_["valid"]["auc"]

            return score_train, score_valid

        scores_train, scores_valid = zip(
            *Parallel(n_jobs=5)(
                delayed(train_fold)(fold) for fold in cv.split(X, y)
            )
        )
        score_train_mean = np.mean(scores_train)
        score_valid_mean = np.mean(scores_valid)

        return score_train_mean, score_valid_mean

    def tuner(self):
        # TODO: add plot of metrics(maybe it's not needed and too complex)
        # TODO: add tracking with MLflow

        sampler = optuna.samplers.TPESampler(seed=8)
        self.study = optuna.create_study(direction="maximize", sampler=sampler)
        self.study.optimize(self.objective_optuna, n_trials=10)

        self.eval_df = pd.DataFrame(
            self.eval_results,
            columns=["trial", "train", "valid", "metric"],
        )

        return self.study

    def final_model(self):
        # TODO: add tracking with MLflow

        final_params = self.study.best_params
        self.final_model = lgb.LGBMClassifier(
            **final_params, random_state=8, importance_type="gain"
        )
        self.final_model.fit(self.df[self.features], self.df[self.target])

        return self.final_model
