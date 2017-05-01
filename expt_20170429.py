import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import xgboost as xgb

tr  = pd.read_csv('train.csv')
tt  = pd.read_csv('test.csv')

tr['dt']  = pd.to_datetime(tr.timestamp)
tt['dt']  = pd.to_datetime(tt.timestamp)
tr['ym']  = tr.timestamp.str[:7]
tt['ym']  = tt.timestamp.str[:7]
tr['price_per_sq']  = tr.price_doc/tr.full_sq

tr['is_investment'] = tr.product_type=='Investment'
tt['is_investment'] = tt.product_type=='Investment'

xgb_params = {'objective':'reg:linear',
              'eval_metric':'rmse',
              'silent':1,
              'booster':'gbtree',
                'eta':0.3,
                'max_depth':3,
                'min_child_weight':1,
                'subsample':0.8,
                'colsample_bytree':1.0,
                'lambda':1,
                'gamma':0,
                'alpha':0}
feat_sets  = [
    ['full_sq','life_sq'],
    ['full_sq','life_sq','build_year','num_room'],
    ['full_sq','life_sq','build_year','num_room','is_investment'],
    ['full_sq','life_sq','build_year','num_room','is_investment','metro_min_walk','metro_km_walk'],
    ['full_sq','life_sq','build_year','num_room','is_investment','metro_min_walk','metro_km_walk','kremlin_km','shopping_centers_km']]
for i,col in enumerate(feat_sets):
    print i,col
    Xtr  = tr[col].fillna(-1)
    Xtt  = tt[col].fillna(-1)
    ytr  = np.log(tr.price_doc + 1)
    dtr  = xgb.DMatrix(Xtr,label=ytr)
    dtt  = xgb.DMatrix(Xtt)
    evals  = {}
    model  = xgb.train(xgb_params,dtr,num_boost_round=20,evals=[(dtr,'tr')],evals_result=evals)
    price_tr_hat  = np.exp(model.predict(dtr)) - 1
    price_tt_hat  = np.exp(model.predict(dtt)) - 1
    print "%.5f" % np.sqrt(np.mean((np.log(tr.price_doc+1)-np.log(price_tr_hat+1))**2))
    tt['price_doc']  = price_tt_hat
    tt[['id','price_doc']].to_csv("submission_20170429_%d.csv" % (i+1),index=False)

feat_rank = pd.concat([
        pd.Series(model.get_score(importance_type='weight')),
        pd.Series(model.get_score(importance_type='gain')),
        pd.Series(model.get_score(importance_type='cover'))],
        1,keys=('wt','gain','cover')).reindex(index=Xtr.columns)
