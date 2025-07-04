import sys
import time
from training import trained_models, test_data, y_mapping

def main():
    if len(sys.argv) != 2:
        raise Exception("Usage: python main.py model")
    model = trained_models[sys.argv[1]]
    X, y = test_data
    for dp_x, y_true in zip(X, y):
        y_pred = model.predict(dp_x)
        print(f'MODEL PREDICTION -> {y_mapping[y_pred]} || THE TRUTH -> {y_mapping[y_true]}')
        print(f'MODEL CORRECT: {y_true == y_pred}')
        time.sleep(1)


if __name__=='__main__':
    main()