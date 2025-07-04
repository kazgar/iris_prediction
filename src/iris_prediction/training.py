from models import Models, run_grid_search
from data_prep import expand_dataset, split_data, split_x_y, transform_y, concat_results, save_params
from param_grids import ParameterGrids
from metric_functions import calculate_basic_metrics
from setup.constants import PROJECT_DATA, PROJECT_ROOT
import polars as pl
from load_model import load_model
from sklearn.exceptions import ConvergenceWarning
from warnings import simplefilter
import os

if not os.path.isdir(PROJECT_ROOT / 'models'):
    os.mkdir(PROJECT_ROOT / 'models')

simplefilter("ignore", category=ConvergenceWarning)

iris_df = pl.read_csv(PROJECT_DATA)
iris_df_expanded = expand_dataset(iris_df)
X, y = split_x_y(iris_df_expanded)
y, y_mapping = transform_y(y)
results = {}

X_train, X_val, X_test, y_train, y_val, y_test = split_data(
    X, y, train_size=0.8, val_set=True
)

lr_best_params = run_grid_search(
    estimator=Models().logistic_regression(),
    param_grid=ParameterGrids().logistic_regression,
    X=X_train,
    y=y_train
)
save_params(lr_best_params, PROJECT_ROOT / 'models/lr.csv')
lr = load_model('lr').fit(X_train, y_train)
results['Logistic Regression'] = pl.from_dict(
    calculate_basic_metrics(y_val, lr.predict(X_val), 'Logistic Regression')
)

svc_best_params = run_grid_search(
    estimator=Models().svc(),
    param_grid=ParameterGrids().svc,
    X=X_train,
    y=y_train
)
save_params(svc_best_params, PROJECT_ROOT / 'models/svc.csv')
svc = load_model('svc').fit(X_train, y_train)
results['SVC'] = pl.from_dict(
    calculate_basic_metrics(y_val, svc.predict(X_val), 'SVC')
)

lsvc_best_params = run_grid_search(
    estimator=Models().linear_svc(),
    param_grid=ParameterGrids().linear_svc,
    X=X_train,
    y=y_train
)
save_params(lsvc_best_params, PROJECT_ROOT / 'models/lsvc.csv')
lsvc = load_model('lsvc').fit(X_train, y_train)
results['Linear SVC'] = pl.from_dict(
    calculate_basic_metrics(y_val, lsvc.predict(X_val), 'Linear SVC')
)

gbc_best_params = run_grid_search(
    estimator=Models().gbc(),
    param_grid=ParameterGrids().gbc,
    X=X_train,
    y=y_train
)
save_params(gbc_best_params, PROJECT_ROOT / 'models/gbc.csv')
gbc = load_model('gbc').fit(X_train, y_train)
results['Gradient Boosting Classifier'] = pl.from_dict(
    calculate_basic_metrics(y_val, gbc.predict(X_val), 'Gradient Boosting Classifier')
)

hgbc_best_params = run_grid_search(
    estimator=Models().hgbc(),
    param_grid=ParameterGrids().hgbc,
    X=X_train,
    y=y_train
)
save_params(hgbc_best_params, PROJECT_ROOT / 'models/hgbc.csv')
hgbc = load_model('hgbc').fit(X_train, y_train)
results['Hist Gradient Boosting Classifier'] = pl.from_dict(
    calculate_basic_metrics(y_val, hgbc.predict(X_val), 'Hist Gradient Boosting Classifier')
)

rf_best_params = run_grid_search(
    estimator=Models().rf(),
    param_grid=ParameterGrids().rf,
    X=X_train,
    y=y_train
)
save_params(rf_best_params, PROJECT_ROOT / 'models/rf.csv')
rf = load_model('rf').fit(X_train, y_train)
results['Random Forest'] = pl.from_dict(
    calculate_basic_metrics(y_val, rf.predict(X_val), 'Random Forest')
)

trained_models = {
    'lr': lr,
    'svc': svc,
    'lsvc': lsvc,
    'gbc': gbc,
    'hgbc': hgbc,
    'rf': rf
}
test_data = [X_test, y_test]

overall_results = concat_results(results)
print('OVERALL RESULTS:')
print('---------------------------------------')
print(overall_results)
