# -*- coding: utf-8 -*-
"""
@author: quincyqiang
@software: PyCharm
@file: lgb.py
@time: 2020/9/2 23:33
@description：
"""
import time
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold
from gen_feas import load_data
from utils import label_inverse


def train_model_classification(X, X_test, y, params, num_classes=2,
                               folds=None, model_type='lgb',
                               eval_metric='logloss', columns=None,
                               plot_feature_importance=False,
                               model=None, verbose=10000,
                               early_stopping_rounds=200,
                               splits=None, n_folds=3):
    """
    分类模型函数
    返回字典，包括： oof predictions, test predictions, scores and, if necessary, feature importances.
    :params: X - 训练数据， pd.DataFrame
    :params: X_test - 测试数据，pd.DataFrame
    :params: y - 目标
    :params: folds - folds to split data
    :params: model_type - 模型
    :params: eval_metric - 评价指标
    :params: columns - 特征列
    :params: plot_feature_importance - 是否展示特征重要性
    :params: model - sklearn model, works only for "sklearn" model type
    """
    start_time = time.time()
    global y_pred_valid, y_pred

    columns = X.columns if columns is None else columns
    X_test = X_test[columns]
    splits = folds.split(X, y) if splits is None else splits
    n_splits = folds.n_splits if splits is None else n_folds

    # to set up scoring parameters
    metrics_dict = {
        'logloss': {
            'lgb_metric_name': 'logloss',
            'xgb_metric_name': 'mlogloss',
            'catboost_metric_name': 'Logloss',
            'sklearn_scoring_function': metrics.log_loss
        },
        'lb_score_method': {
            'sklearn_scoring_f1': metrics.f1_score,  # 线上评价指标
            'sklearn_scoring_accuracy': metrics.accuracy_score,  # 线上评价指标
            'sklearn_scoring_auc': metrics.roc_auc_score
        },
    }
    result_dict = {}

    # out-of-fold predictions on train data
    oof = np.zeros(shape=(len(X), num_classes))
    # averaged predictions on train data
    prediction = np.zeros(shape=(len(X_test), num_classes))
    # list of scores on folds
    scores = []
    # feature importance
    feature_importance = pd.DataFrame()

    # split and train on folds
    for fold_n, (train_index, valid_index) in enumerate(splits):
        if verbose:
            print(f'Fold {fold_n + 1} started at {time.ctime()}')
        if type(X) == np.ndarray:
            X_train, X_valid = X[train_index], X[valid_index]
            y_train, y_valid = y[train_index], y[valid_index]
        else:
            X_train, X_valid = X[columns].iloc[train_index], X[columns].iloc[valid_index]
            y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]

        if model_type == 'lgb':
            model = lgb.LGBMClassifier(**params)
            model.fit(X_train, y_train,
                      eval_set=[(X_train, y_train), (X_valid, y_valid)],
                      eval_metric=metrics_dict[eval_metric]['lgb_metric_name'],
                      verbose=verbose,
                      early_stopping_rounds=early_stopping_rounds)

            y_pred_valid = model.predict_proba(X_valid)
            y_pred = model.predict_proba(X_test, num_iteration=model.best_iteration_)

        if model_type == 'xgb':
            model = xgb.XGBClassifier(**params)
            model.fit(X_train, y_train,
                      eval_set=[(X_train, y_train), (X_valid, y_valid)],
                      eval_metric=metrics_dict[eval_metric]['xgb_metric_name'],
                      verbose=bool(verbose),  # xgb verbose bool
                      early_stopping_rounds=early_stopping_rounds)
            y_pred_valid = model.predict_proba(X_valid)
            y_pred = model.predict_proba(X_test, ntree_limit=model.best_ntree_limit)
        if model_type == 'sklearn':
            model = model
            model.fit(X_train, y_train)
            y_pred_valid = model.predict_proba(X_valid)
            score = metrics_dict[eval_metric]['sklearn_scoring_function'](y_valid, y_pred_valid)
            print(f'Fold {fold_n}. {eval_metric}: {score:.4f}.')
            y_pred = model.predict_proba(X_test)

        if model_type == 'cat':
            model = CatBoostClassifier(iterations=20000, eval_metric=metrics_dict[eval_metric]['catboost_metric_name'],
                                       **params,
                                       loss_function=metrics_dict[eval_metric]['catboost_metric_name'])
            model.fit(X_train, y_train, eval_set=(X_valid, y_valid), cat_features=[], use_best_model=True,
                      verbose=False)

            y_pred_valid = model.predict_proba(X_valid)
            y_pred = model.predict_proba(X_test)

        oof[valid_index] = y_pred_valid
        # 评价指标
        scores.append(
            metrics_dict['lb_score_method']['sklearn_scoring_accuracy'](y_valid, np.argmax(y_pred_valid, axis=1)))
        print(scores)
        prediction += y_pred

        if model_type == 'lgb' and plot_feature_importance:
            # feature importance
            fold_importance = pd.DataFrame()
            fold_importance["feature"] = columns
            fold_importance["importance"] = model.feature_importances_
            fold_importance["fold"] = fold_n + 1
            feature_importance = pd.concat([feature_importance, fold_importance], axis=0)

        if model_type == 'xgb' and plot_feature_importance:
            # feature importance
            fold_importance = pd.DataFrame()
            fold_importance["feature"] = columns
            fold_importance["importance"] = model.feature_importances_
            fold_importance["fold"] = fold_n + 1
            feature_importance = pd.concat([feature_importance, fold_importance], axis=0)
    prediction /= n_splits
    print('CV mean score: {0:.4f}, std: {1:.4f}.'.format(np.mean(scores), np.std(scores)))

    result_dict['oof'] = oof
    result_dict['prediction'] = prediction
    result_dict['scores'] = scores

    if model_type == 'lgb' or model_type == 'xgb':
        if plot_feature_importance:
            feature_importance["importance"] /= n_splits
            cols = feature_importance[["feature", "importance"]].groupby("feature").mean().sort_values(
                by="importance", ascending=False)[:50].index

            best_features = feature_importance.loc[feature_importance.feature.isin(cols)]

            plt.figure(figsize=(16, 12))
            sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False))
            plt.title('LGB Features (avg over folds)')
            plt.show()
            result_dict['feature_importance'] = feature_importance
            print(feature_importance)
    end_time = time.time()

    print("train_model_classification cost time:{}".format(end_time - start_time))
    return result_dict


lgb_params = {
    'boosting_type': 'gbdt',
    'objective': 'multiclass',
    'n_estimators': 100000,
    'learning_rate': 0.1,
    'random_state': 2948,
    'bagging_freq': 8,
    'bagging_fraction': 0.80718,
    # 'bagging_seed': 11,
    'feature_fraction': 0.38691,  # 0.3
    'feature_fraction_seed': 11,
    'max_depth': 9,
    'min_data_in_leaf': 40,
    'min_child_weight': 0.18654,
    "min_split_gain": 0.35079,
    'min_sum_hessian_in_leaf': 1.11347,
    'num_leaves': 29,
    'num_threads': 4,
    "lambda_l1": 0.55831,
    'lambda_l2': 1.67906,
    'cat_smooth': 10.4,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'n_jobs': -1
    # 'verbosity': 1,
}

n_fold = 5
nums = [3, 4, 5, 11]
for num_classes in nums:
    print("分类个数num_classes:{}".format(num_classes))
    folds = StratifiedKFold(n_splits=n_fold, random_state=1314)
    train, _, test, features = load_data(num_classes)

    X = train[features]
    X['Year-Of-Publication'] = X['Year-Of-Publication'].astype(int)
    X_test = test[features]
    X_test['Year-Of-Publication'] = X_test['Year-Of-Publication'].astype(int)
    y = train['Book-Rating'].astype(int)
    print(y.value_counts())
    X = train[features]
    X_test = test[features]

    result_dict_lgb = train_model_classification(X=X,
                                                 X_test=X_test,
                                                 y=y,
                                                 params=lgb_params,
                                                 num_classes=num_classes,
                                                 folds=folds,
                                                 model_type='lgb',
                                                 eval_metric='logloss',
                                                 plot_feature_importance=True,
                                                 verbose=10,
                                                 early_stopping_rounds=200)

    pred = np.argmax(result_dict_lgb['prediction'], axis=1)
    test['Book-Rating'] = pred

    test['Book-Rating'] = test['Book-Rating'].apply(lambda x: label_inverse(x, num_classes))
    score = np.mean(result_dict_lgb['scores'])
    print(score)
    test['id'] = test['id'].astype(int)
    test[['id', 'Book-Rating']].to_csv('result/lgb_numclasses{}_{}.csv'.format(num_classes, score), index=False,
                                       header=False)
    print(test['Book-Rating'].value_counts())

    pd.DataFrame(result_dict_lgb['oof']).to_csv('result/lgb_numclasses{}_{}trainoof.csv'.format(num_classes, score),
                                                index=False, header=False)
    pd.DataFrame(result_dict_lgb['prediction']).to_csv(
        'result/lgb_numclasses{}_{}testoof.csv'.format(num_classes, score),
        index=False, header=False)
