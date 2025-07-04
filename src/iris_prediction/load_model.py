from setup.constants import PROJECT_ROOT
import polars as pl
from models import Models

def load_model(
        model_name:str
):
    params = pl.read_csv(PROJECT_ROOT / 'models' / f'{model_name}.csv')
    if model_name == 'lr':
        model = Models().logistic_regression(
            C=params['C'][0],
            max_iter=params['max_iter'][0],
            penalty=params['penalty'][0],
            solver=params['solver'][0],
            tol=params['tol'][0]
        )
    
    elif model_name == 'svc':
        model = Models().svc(
            C=params["C"][0],
            degree=params["degree"][0],
            gamma=params["gamma"][0],
            kernel=params["kernel"][0],
            max_iter=params["max_iter"][0],
        )    
    
    elif model_name == 'lsvc':
        model = Models().linear_svc(
            C=params["C"][0],
            loss=params["loss"][0],
            max_iter=params["max_iter"][0],
            multi_class=params["multi_class"][0],
        )
    
    elif model_name == 'gbc':
        model = Models().gbc(
            learning_rate=params["learning_rate"][0],
            loss=params["loss"][0],
            max_depth=params["max_depth"][0],
            n_estimators=params["n_estimators"][0],
        )
    
    
    elif model_name == 'hgbc':
        model = Models().hgbc(
            learning_rate=params["learning_rate"][0],
            max_iter=params["max_iter"][0],
            max_depth=params["max_depth"][0],
        )

    elif model_name == 'rf':
        model = Models().rf(
            criterion=params["criterion"][0],
            max_depth=params["max_depth"][0],
            max_features=params["max_features"][0],
            n_estimators=params["n_estimators"][0],
        )

    return model