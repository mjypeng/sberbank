from common import *
import xgboost as xgb

def submit_expt(expt,tr,tt,col,name):
    dtr     = dataframe_to_dmatrix(tr,col)
    dtt     = dataframe_to_dmatrix(tt,col)
    for i,idx in enumerate(expt.index):
        xgb_params  = idx_to_xgb_params(idx,expt.index.names)
        #
        model  = xgb.train(xgb_params,dtr,num_boost_round=expt.loc[idx,'optn']) #.astype(int)
        price_tr_hat  = np.exp(model.predict(dtr)) - 1
        price_tt_hat  = np.exp(model.predict(dtt)) - 1
        if np.all(expt.loc[idx,['tr','avg log pred']].values - [np.sqrt(np.mean((np.log1p(tr.price_doc)-np.log1p(price_tr_hat))**2)),np.mean(np.log1p(price_tt_hat))] < 10**-7): print "OK"
        else: print "metric mismatch!!!"
        #
        tt['price_doc']  = price_tt_hat
        submit(tt,"%s_%d_%s" % (name,i+1,str(idx)))

def xgb_cross_validation(xgb_params,tr,col,target='price_doc'):
    dtr       = dataframe_to_dmatrix(tr,col,target=target)
    cv_evals  = xgb.cv(xgb_params,dtr,num_boost_round=1000,nfold=5,metrics=('rmse'),early_stopping_rounds=xgb_params['early_stop'],verbose_eval=True,show_stdv=True,seed=0,shuffle=False)
    opt_nboost  = len(res)
    cv_evals.rename(columns={'train-rmse-mean':'tr1', 'train-rmse-std':'tr1_std', 'test-rmse-mean': 'vn', 'test-rmse-std':'vn_std'},inplace=True)
    return opt_nboost,cv_evals

def xgb_temporal_validation(xgb_params,tr,col,target='price_doc'):
    baseline_rmsle = np.sqrt(np.mean((np.log(tr.loc[tr.timestamp>=xgb_params['vn_part'],target]+1) - np.log(tr.loc[tr.timestamp<xgb_params['vn_part'],target]+1).mean())**2))
    print "Baseline RMSLE = %.5f" % baseline_rmsle
    dtr  = dataframe_to_dmatrix(tr[tr.timestamp<xgb_params['vn_part']],col,target=target)
    dvn  = dataframe_to_dmatrix(tr[tr.timestamp>=xgb_params['vn_part']],col,target=target)
    tv_evals  = {}
    tv_model  = xgb.train(xgb_params,dtr,num_boost_round=1000,evals=[(dtr,'tr1'),(dvn,'vn')],early_stopping_rounds=xgb_params['early_stop'],evals_result=tv_evals,verbose_eval=False)
    opt_nboost = tv_model.best_ntree_limit
    print "Training RMSLE = %.5f" % tv_evals['tr1']['rmse'][opt_nboost-1]
    print "Validation RMSLE = %.5F" % tv_evals['vn']['rmse'][opt_nboost-1]
    tv_evals  = pd.DataFrame(zip(tv_evals['tr1']['rmse'][:opt_nboost],tv_evals['vn']['rmse'][:opt_nboost]),columns=('tr1','vn'))
    return opt_nboost,tv_evals

def idx_to_xgb_params(idx,names):
    xgb_params = {'objective':'reg:linear',
                  'eval_metric':'rmse',
                  'silent':1,
                  'booster':'gbtree',
                    'eta':0.3,
                    'max_depth':6,
                    'min_child_weight':1,
                    'subsample':0.8,
                    'colsample_bytree':1.0,
                    'lambda':1,
                    'gamma':0,
                    'alpha':0}
    for value,name in zip(idx,names):
        xgb_params[name]  = value
    return xgb_params

def xgb_feat_rank(model,col):
    feat_rank  = pd.concat([
        pd.Series(model.get_score(importance_type='weight')),
        pd.Series(model.get_score(importance_type='gain')),
        pd.Series(model.get_score(importance_type='cover'))],
        1,keys=('wt','gain','cover')).reindex(index=col)
    return feat_rank

def dataframe_to_dmatrix(d,col,target='price_doc'):
    return xgb.DMatrix(d[col].fillna(-1),label=np.log(d[target] + 1) if target in d else None)
