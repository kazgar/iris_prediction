import sys
import time
import polars as pl
from setup.constants import PROJECT_DATA
from data_prep import expand_dataset, split_x_y, transform_y, split_data
from load_model import load_model

def main():
    if len(sys.argv) != 2:
        raise Exception("Usage: python main.py model")
    iris_df = pl.read_csv(PROJECT_DATA)
    iris_df_expanded = expand_dataset(iris_df)
    X, y = split_x_y(iris_df_expanded)
    y, y_mapping = transform_y(y)
    y_mapping = {nr: nm for nm, nr in y_mapping.items()}
    print(y_mapping)
    X_train, _, X_test, y_train, _, y_test = split_data(
    X, y, train_size=0.8, val_set=True
    )
    model = load_model(sys.argv[1]).fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print('\n')
    for y_p, y_t in zip(y_pred, y_test):
        print(f'Model prediction -> {y_mapping[y_p]} || The truth: {y_mapping[y_t]} || VERDICT: {y_p == y_t}\n')
        time.sleep(1)

if __name__=='__main__':
    main()