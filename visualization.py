import operator

import pandas as pd
import xgboost as xgb

def get_feature_importances(clf, df_train):
    importances = {'feature': [], 'importance': []}
    f_importances = clf.feature_importances_
    for col, importance in zip(df_train.columns, f_importances):
        importances['feature'].append(col)
        importances['importance'].append(importance)
    df_importance = pd.DataFrame(importances)
    return df_importance


def create_feature_map(features):
    outfile = open('/tmp/xgb.fmap', 'w')
    i = 0
    for feat in features:
        outfile.write('{0}\t{1}\tq\n'.format(i, feat))
        i = i + 1

    outfile.close()


def get_xgr_importances(df_train, y_train):
    create_feature_map(df_train.columns)
    xgb_params = {"objective": "reg:linear", "eta": 0.05, "max_depth": 10,
                  "seed": 2016, "silent": 1, "n_estimators": 100,
                   "min_child_weight": 1.5, "colsample_bytree": 0.5}
    num_rounds = 1000
    dtrain = xgb.DMatrix(df_train.values, label=y_train)
    gbdt = xgb.train(xgb_params, dtrain, num_rounds)

    importance = gbdt.get_fscore(fmap='/tmp/xgb.fmap')
    importance = sorted(importance.items(), key=operator.itemgetter(1))

    df_xgb_imp = pd.DataFrame(importance, columns=['feature', 'importance'])
    df_xgb_imp['importance'] = df_xgb_imp['importance'] / df_xgb_imp['importance'].sum()
    return df_xgb_imp
