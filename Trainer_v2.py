import optuna
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import mlflow


class Trainer:
    def __init__(self, df_train, df_test, features, target):
        self.df = df_train
        self.df_test = df_test
        self.features = features
        self.target = target
        self.eval_results = []
        self.eval_df = None

    def custom_metric(self, train, valid):
        alpha = 3
        beta = 0.1
        return alpha * valid - beta * abs(train - valid)

    def gini(self, y_true, y_pred):
        return 2 * roc_auc_score(y_true, y_pred) - 1

    def auc(self, y_true, y_pred):
        return roc_auc_score(y_true, y_pred)

    def objective_optuna(self, trial):
        params = {
            "objective": "binary",
            "metric": "auc",
            "verbosity": -1,
            "boosting_type": "gbdt",
            "max_depth": trial.suggest_int("max_depth", 2, 3),
            "learning_rate": trial.suggest_float(
                "learning_rate", 0.01, 0.1, log=True
            ),
            "n_estimators": 20,
            "num_leaves": trial.suggest_int("num_leaves", 2, 10),
            "max_bin": trial.suggest_int("max_bin", 50, 250),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.01, 5.0, log=True),
            "reg_lambda": trial.suggest_float(
                "reg_lambda", 0.01, 5.0, log=True
            ),
            "min_child_samples": trial.suggest_int(
                "min_child_samples", 200, 2000
            ),
        }

        X = self.df[self.features]
        y = self.df[self.target]

        model = lgb.LGBMClassifier(
            **params, random_state=8, importance_type="gain"
        )

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=8)
        mod_callbacks = [
            lgb.early_stopping(3, min_delta=0.001),
            lgb.log_evaluation(period=100),
            optuna.integration.LightGBMPruningCallback(
                trial, "auc", valid_name="valid"
            ),
        ]

        score_train_cv = np.empty(5)
        score_valid_cv = np.empty(5)

        for idx, (train_idx, valid_idx) in enumerate(cv.split(X, y)):
            X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
            X_valid, y_valid = X.iloc[valid_idx], y.iloc[valid_idx]

            model.fit(
                X_train,
                y_train,
                eval_set=[(X_train, y_train), (X_valid, y_valid)],
                eval_names=["train", "valid"],
                eval_metric="auc",
                callbacks=mod_callbacks,
            )

            score_train_cv[idx] = model.best_score_["train"]["auc"]
            score_valid_cv[idx] = model.best_score_["valid"]["auc"]

        score_train = np.mean(score_train_cv)
        score_valid = np.mean(score_valid_cv)

        metric = self.custom_metric(score_train, score_valid)
        self.eval_results.append(
            [trial.number, score_train, score_valid, metric]
        )

        return metric

    def tuner(self, n_trials=10, name=None, tracking=False, tracking_uri=None):
        # TODO: add plot of metrics(maybe it's not needed and too complex)
        # TODO: add tracking with MLflow

        if tracking:
            mlflc = optuna.integration.MLflowCallback(
                tracking_uri=tracking_uri,
                metric_name="metric",
            )
        else:
            mlflc = None

        optuna.logging.set_verbosity(optuna.logging.CRITICAL)
        sampler = optuna.samplers.TPESampler(seed=8)
        self.study = optuna.create_study(
            study_name=name,
            direction="maximize",
            sampler=sampler,
            pruner=optuna.pruners.HyperbandPruner(),
            load_if_exists=True,
        )
        self.study.optimize(
            self.objective_optuna,
            n_trials=n_trials,
            show_progress_bar=True,
            callbacks=[mlflc],
        )

        self.eval_df = pd.DataFrame(
            self.eval_results,
            columns=["trial", "train", "valid", "metric"],
        )

        return self.study

    def train_final_model(self, model):
        tr_model = model.fit(self.df[self.features], self.df[self.target])

        proba_train = tr_model.predict_proba(self.df[self.features])[:, 1]
        proba_test = tr_model.predict_proba(self.df_test[self.features])[:, 1]

        return tr_model, proba_train, proba_test

    def final_model(
        self, name=None, run_name=None, tracking=False, tracking_uri=None
    ):
        # TODO: add tracking with MLflow

        final_params = self.study.best_params
        self.final_model = lgb.LGBMClassifier(
            **final_params, random_state=8, importance_type="gain"
        )

        if tracking:
            mlflow.set_tracking_uri(tracking_uri)
            mlflow.set_experiment(name)

            with mlflow.start_run():
                for key, value in final_params.items():
                    mlflow.log_param(key, value)

                (
                    self.final_model,
                    proba_train,
                    proba_test,
                ) = self.train_final_model(self.final_model)

                self.gini_train = self.gini(self.df[self.target], proba_train)
                self.gini_test = self.gini(
                    self.df_test[self.target], proba_test
                )
                self.metric = self.custom_metric(
                    self.auc(self.df[self.target], proba_train),
                    self.auc(self.df_test[self.target], proba_test),
                )

                mlflow.log_metric("gini_train", self.gini_train)
                mlflow.log_metric("gini_test", self.gini_test)
                mlflow.log_metric("metric", self.metric)

                mlflow.set_tag("mlflow.runName", run_name)

        else:
            self.final_model, proba_train, proba_test = self.train_final_model(
                self.final_model
            )

            self.gini_train = self.gini(self.df[self.target], proba_train)
            self.gini_test = self.gini(self.df_test[self.target], proba_test)
            self.metric = self.custom_metric(
                self.auc(self.df[self.target], proba_train),
                self.auc(self.df_test[self.target], proba_test),
            )

        return self.final_model
