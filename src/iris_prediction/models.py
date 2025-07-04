from sklearn.ensemble import (
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier
)
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import LogisticRegression
from enum import Enum

class Models(Enum):
    logistic_regression=LogisticRegression()
    svc=SVC()
    linear_svc=LinearSVC()
    gbc=GradientBoostingClassifier()
    hgbc=HistGradientBoostingClassifier()
    rf=RandomForestClassifier()
