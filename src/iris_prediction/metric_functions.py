from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def calculate_basic_metrics(y_true, y_pred, model_name) -> dict:
    return {
        "model_name": model_name,
        "accuracy": accuracy_score(y_true, y_pred),
        "precission": precision_score(y_true, y_pred, average="weighted"),
        "recall": recall_score(y_true, y_pred, average="weighted"),
        "f1": f1_score(y_true, y_pred, average="weighted"),
    }
