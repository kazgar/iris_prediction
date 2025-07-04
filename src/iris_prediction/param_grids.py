class ParameterGrids():
    def __init__(self):    
        self.logistic_regression = {
            "penalty": ["l2"],
            "solver": ["lbfgs", "newton-cg", "newton-cholesky", "sag", "saga"],
            "C": [0.0001, 0.001, 0.01, 0.1, 1],
            "tol": [0.00001, 0.0001, 0.001, 0.01],
            "max_iter": [100, 300, 500],
        }
        self.svc = {
            "C": [0.0001, 0.001, 0.01, 0.1, 1],
            "kernel": ["poly", "rbf", "sigmoid"],
            "degree": [1, 2, 3, 4],
            "gamma": ["scale", "auto"],
            "max_iter": [100, 300, 500],
        }
        self.linear_svc = {
            "loss": ["hinge", "squared_hinge"],
            "C": [0.0001, 0.001, 0.01, 0.1, 1],
            "multi_class": ["ovr", "crammer_singer"],
            "max_iter": [100, 300, 500],
        }
        self.gbc = {
            "loss": ["log_loss"],
            "learning_rate": [0.0001, 0.001, 0.01, 0.1, 1],
            "n_estimators": [1, 10, 100, 1000],
            "max_depth": [1, 3, 5],
        }
        self.hgbc = {
            "learning_rate": [0.0001, 0.001, 0.01, 0.1, 1],
            "max_iter": [100, 300, 500],
            "max_depth": [1, 3, 5, 7],
        }
        self.rf = {
            "n_estimators": [1, 3, 5, 10, 100],
            "criterion": ["gini", "entropy", "log_loss"],
            "max_depth": [3, 5, 7, 10, None],
            "max_features": ["sqrt", "log2", None],
        }