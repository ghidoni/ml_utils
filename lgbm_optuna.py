import optuna
import lightgbm as lgb
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from typing import List, Optional
from sklearn.model_selection import cross_validate


def custom_metric(train, valid):
    alpha = 3
    beta = 0.1
    return alpha * valid - beta * abs(train - valid)


def objective(trial):
    # define hyperparameters to be tuned
    params = {
        "num_leaves": trial.suggest_int("num_leaves", 2, 100),
        "learning_rate": trial.suggest_loguniform("learning_rate", 0.001, 0.1),
        "n_estimators": 1000,
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "subsample": trial.suggest_uniform("subsample", 0.1, 1),
        "colsample_bytree": trial.suggest_uniform("colsample_bytree", 0.1, 1),
        "reg_alpha": trial.suggest_loguniform("reg_alpha", 1e-5, 10),
        "reg_lambda": trial.suggest_loguniform("reg_lambda", 1e-5, 10),
        "random_state": 42,
        "n_jobs": -1,
    }

    pruning_callback = optuna.integration.LightGBMPruningCallback(trial, "auc")

    # define the model
    model = lgb.LGBMClassifier(**params)

    # train the model
    # model.fit(X_train, y_train,
    #            callbacks=[lgb.early_stopping(10),lgb.log_evaluation(100),pruning_callback])
    cross_results = cross_validate(
        model,
        X_train,
        y_train,
        cv=5,
        scoring="roc_auc",
        n_jobs=-1,
        return_train_score=True,
        fit_params={
            "eval_set": [(X_train, y_train), (X_val, y_val)],
            "eval_metric": "auc",
            "callbacks": [
                lgb.early_stopping(10),
                lgb.log_evaluation(100),
                pruning_callback,
            ],
        },
    )

    # evaluate the model
    result = custom_metric(
        cross_results["train_score"].mean(), cross_results["test_score"].mean()
    )

    # return the objective value to be maximized
    return result


study = optuna.create_study(
    direction="maximize", pruner=optuna.pruners.MedianPruner(n_warmup_steps=10)
)
study.optimize(objective, n_trials=10)
