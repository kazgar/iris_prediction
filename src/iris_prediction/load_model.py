from setup.constants import PROJECT_ROOT
import polars as pl
from models import Models

def load_model(
        model_name:str
):
    params = pl.read_csv(PROJECT_ROOT / 'models' / f'{model_name}.csv')
    if model_name == 'lr':
        model = Models().logistic_regression(
            C=params['C'],
            max_iter=params['max_iter'],
            penalty=params['penalty'],
            solver=params['solver'],
            tol=params['tol']
        )
    
    elif model_name == 'svc':
        model = Models().svc(
            C=params["C"],
            degree=params["degree"],
            gamma=params["gamma"],
            kernel=params["kernel"],
            max_iter=params["max_iter"],
        )    
    
    elif model_name == 'lsvc':
        model = Models().lsvc(
            C=params["C"],
            loss=params["loss"],
            max_iter=params["max_iter"],
            multi_class=params["multi_class"],
        )
    
    elif model_name == 'gbc':
        model = Models().gbc(
            learning_rate=params["learning_rate"],
            loss=params["loss"],
            max_depth=params["max_depth"],
            n_estimators=params["n_estimators"],
        )
    
    
    elif model_name == 'hgbc':
        model = Models().hgbc(
            learning_rate=params["learning_rate"],
            max_iter=params["max_iter"],
            max_depth=params["max_depth"],
        )

    elif model_name == 'rf':
        model = Models().rf(
            criterion=params["criterion"],
            max_depth=params["max_depth"],
            max_features=params["max_features"],
            n_estimators=params["n_estimators"],
        )
        
    return model