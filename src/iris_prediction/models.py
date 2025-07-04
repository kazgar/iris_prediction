from sklearn.ensemble import (
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
import polars as pl


class Models():
    def __init__(self):
        self.logistic_regression = LogisticRegression
        self.svc = SVC
        self.linear_svc = LinearSVC
        self.gbc = GradientBoostingClassifier
        self.hgbc = HistGradientBoostingClassifier
        self.rf = RandomForestClassifier

def run_grid_search(
    estimator,
    param_grid: dict,
    X: pl.DataFrame,
    y: pl.Series | pl.DataFrame,
    scoring: str = 'accuracy',
    refit: bool = True
) -> dict:
    params = (
        GridSearchCV(
            estimator=estimator, param_grid=param_grid, scoring=scoring, refit=refit
        )
        .fit(X, y)
        .best_params_
    )
    return params
