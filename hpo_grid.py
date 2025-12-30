"""
Configuration file for hyperparameter grids used in model tuning.
References:
https://arxiv.org/pdf/2505.14415 (LASSO, Ridge, RF, XGBoost, CatBoost)
https://arxiv.org/pdf/2207.08815 (Gradient Boosting)
https://arxiv.org/pdf/2407.04491 (LightGBM)
"""
import numpy as np


PARAM_GRIDS = {
    'Logistic': {
        'lr__penalty': [None],
    },

    'LASSO': { 
        'lr__penalty': ["l1"],
        'lr__solver': ["liblinear"],
        'lr__C': [0.01, 0.1, 1, 10, 100],
    },

    'Ridge': { 
        'lr__penalty': ['l2'],
        'lr__C': [0.01, 0.1, 1, 10, 100],
    },

    'Random Forest':{ 
        'n_estimators': np.arange(50,300, 50), # randint(50,250),
        'max_depth': [None, 2, 3, 4],
        'max_features': ["sqrt","log2",None, 0.2, 0.4, 0.6, 0.8], # ["sqrt","log2",None, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        'min_samples_leaf': list(np.round(np.logspace(np.log10(1.5), np.log10(50.5), num=5).astype(int))), # loguniform(1.5, 50.5),
        'bootstrap': [True, False],
        'min_impurity_decrease': [0, 0.01, 0.02, 0.05],
        'n_jobs': [1],
    },

    'Gradient Boosting': {  
        'gb__learning_rate': list((np.round(np.logspace(np.log10(1e-5), np.log10(1), num=6),5))), # lognorm(np.log(0.01), np.log(10)), 
        'gb__subsample': [0.5, 0.8, 1], # uniform(0.5, 1),
        'gb__n_estimators': [1000], # loguniform_int(10.5, 1000.5),
        'gb__max_depth': [None, 2, 3, 4, 5], #(with weight = (0.1, 0.1, 0.6, 0.1, 0.1))
        'gb__min_samples_split': [2, 3], #(with weight = (0.95, 0.05))
        'gb__min_samples_leaf': list(np.round(np.logspace(np.log10(1.5), np.log10(50.5), num=5).astype(int))), # loguniform_int(1.5, 50.5),
        'gb__min_impurity_decrease': [0, 0.01, 0.02, 0.05], #(with weight = (0.85, 0.5, 0.5, 0.5)
        'gb__max_leaf_nodes': [None, 5, 10, 15], #(with weight =(0.85, 0.5, 0.5, 0.5)),
        'gb__validation_fraction': [0.1],
        'gb__n_iter_no_change': [50],
    },

    'XGBoost': {  
        'n_estimators': [1000],
        'max_depth': np.arange(2,11,2), # randint(2,10),
        'learning_rate': list((np.round(np.logspace(np.log10(1e-5), np.log10(1), num=6),5))), #loguniform(0.00001, 1),
        'min_child_weight': list((np.round(np.logspace(np.log10(1), np.log10(100), num=6),5))), #loguniform(1, 100),
        'subsample': [0.5, 0.8, 1], #uniform(0.5, 1),
        'colsample_bylevel': [0.5, 0.8, 1],# uniform(0.5, 1),
        'colsample_bytree': [0.5, 0.8, 1], # uniform(0.5, 1),
        'gamma': list((np.round(np.logspace(np.log10(1e-8), np.log10(7), num=6),5))), #loguniform(0.00000001, 7),
        'lambda': list((np.round(np.logspace(np.log10(1), np.log10(4), num=4),3))), # loguniform(1, 4),
        'alpha': list((np.round(np.logspace(np.log10(1e-8), np.log10(100), num=5),7))), # loguniform(0.00000001, 100),
        'early_stopping_rounds': [50], # doesn't work for cross validation, early stopping rounds is more suited of train/val/test split
    #   'verbose': [False],   
    #   'tree_method': ['hist'],
    #   'device': ['cuda']
    },

    'LightGBM':{ 
        'n_estimators': [1000],
        'bagging_freq': [1],
        'num_leaves':  [2, 10, 100, 1000, 10000], #list((np.round(np.logspace(np.log10(1), np.log10(1e5), num=6).astype(int)))), # loguniform_int(1,np.exp(1)**7), the documentation says that the number of leaves is bounded by 131072
        'learning_rate': list((np.round(np.logspace(np.log10(1e-5), np.log10(1), num=6),5))), # loguniform(np.exp(1)**(-7), 1), in the paper the lower bound is 1e-7 but for consistency with the other boosting method we take 1e-5
        'subsample': [0.5, 0.8, 1], # uniform(0.5, 1),
        'feature_fraction': [0.5, 0.8, 1], # uniform(0.5, 1),
        'min_data_in_leaf': list((np.round(np.logspace(np.log10(1), np.log10(1e5), num=6).astype(int)))), # loguniform_int(1, np.exp(1)**6),
        'min_sum_hessian_in_leaf': list(np.logspace(np.log10(1e-7), np.log10(1e5), num=5)), ## old: list(np.logspace(np.log10(1e-16), np.log10(1e5), num=5)), # oguniform(np.exp(1)**(-16), np.exp(1)**5),
        'lambda_l1': [0, 1e-16, 1e-10, 0.0001, 100.0],
        'lambda_l2': [0, 1e-16, 1e-10, 0.0001, 100.0],
        'verbose': [-1],
        'objective': ['binary'],
        'metric': ['auc'],
       # 'device':['gpu'],
        #'early_stopping_rounds': [300], # doesn't work for cross validation, early stopping rounds is more suited of train/val/test split

    },

    'CatBoost':{ 
        'iterations': [1000],
        'depth': np.arange(2,7,2), # randint(2,6),
        'learning_rate': list((np.round(np.logspace(np.log10(1e-5), np.log10(1), num=6),5))), # loguniform(0.00001, 1),
        'bagging_temperature': [0, 0.25, 0.5, 0.75, 1], # uniform(0,1),
        'l2_leaf_reg':  list((np.round(np.logspace(np.log10(1), np.log10(10), num=4),1))), # loguniform(1 ,10),
        'one_hot_max_size': [0, 5, 15, 25], # randint(0, 25),
        'random_strength': [1, 10, 20], # randint(1, 20),
        'leaf_estimation_iterations': [1, 10, 20], # randint(1, 20),
        'od_wait': [50], # doesn't work for cross validation, early stopping rounds is more suited of train/val/test split
        'od_type': ['Iter'], # doesn't work for cross validation, early stopping rounds is more suited of train/val/test split
        'verbose': [1],
        'eval_metric': ['AUC'],
#        'task_type': ["GPU"]

    },

    'MLP':{
        'mlp__hidden_layer_sizes': [(50), (100), (200)],
        'mlp__activation':['logistic', 'tanh', 'relu'],
        'mlp__learning_rate_init': list((np.round(np.logspace(np.log10(1e-5), np.log10(1), num=6),5))),
        'mlp__early_stopping':[True],
        'mlp__validation_fraction':[0.1],
        'mlp__n_iter_no_change':[50],
        'mlp__max_iter':[500],
    }

}

