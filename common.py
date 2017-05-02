import time
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import xgboost as xgb
import scipy.stats as stats

def xgb_temporal_validation(xgb_params,tr,col):
    dtr  = dataframe_to_dmatrix(tr[tr.timestamp<xgb_params['vn_part_date']],col)
    dvn  = dataframe_to_dmatrix(tr[tr.timestamp>=xgb_params['vn_part_date']],col)
    tv_evals  = {}
    tv_model  = xgb.train(xgb_params,dtr,num_boost_round=1000,evals=[(dtr,'tr1'),(dvn,'vn')],early_stopping_rounds=xgb_params['early_stop'],evals_result=tv_evals,verbose_eval=False)
    opt_nboost = tv_model.best_ntree_limit
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

def xgb_feat_rank(model,col,tr=None):
    feat_rank  = pd.concat([
        pd.Series(model.get_score(importance_type='weight')),
        pd.Series(model.get_score(importance_type='gain')),
        pd.Series(model.get_score(importance_type='cover'))],
        1,keys=('wt','gain','cover')).reindex(index=col)
    return feat_rank

def pearsonr_w_price(d,col,col_price='price_doc',use_log=True):
    mask  = d[col].notnull() & d[col_price].notnull()
    x     = d.loc[mask,col]
    y     = np.log(d.loc[mask,col_price]+1) if use_log else d.loc[mask,col_price]
    return stats.pearsonr(x,y)

def dataframe_to_dmatrix(d,col):
    return xgb.DMatrix(d[col].fillna(-1),label=np.log(d.price_doc + 1) if 'price_doc' in d else None)

def submit(tt,name):
    tt[['id','price_doc']].to_csv("submission_%s.csv" % name,index=False)

def get_dummy_values(d,col,thd):
    c  = d[col].value_counts()
    return c[c>thd].index.tolist()

def preprocess(d):
    d['dt']            = pd.to_datetime(d.timestamp)
    d['ym']            = d.timestamp.str[:7]
    d['year']          = pd.to_numeric(d.ym.str[:4])
    d['month']         = pd.to_numeric(d.ym.str[5:])
    #
    d['sq_inferred']   = np.where(d.full_sq>=10,d.full_sq,np.where(d.life_sq>=10,d.life_sq,np.nan))
    d['life_ratio']    = np.where(d.full_sq>0,d.life_sq/d.full_sq,np.nan)
    d['floor_ratio']   = np.where(d.max_floor>0,d.floor/d.max_floor,np.nan)
    d['build_age']     = np.where((d.build_year>1000)&(d.build_year<3000),d.year-d.build_year,np.nan)
    d['kitch_ratio']   = d.kitch_sq / d.full_sq
    d['is_investment'] = d.product_type=='Investment'
    #
    d['monthly_volume']  = d.groupby('ym').id.count()[d.ym].values
    d['monthly_volume_invest']  = d[d.is_investment].groupby('ym').id.count()[d.ym].fillna(0).values
    d['monthly_volume_occupy']  = d[~d.is_investment].groupby('ym').id.count()[d.ym].fillna(0).values
    #
    d['sub_area_pop_density'] = d.full_all / d.area_m
    #
    d['sub_area_volume']  = d.groupby('sub_area').id.count()[d.sub_area].values
    d['sub_area_volume_invest']  = d[d.is_investment].groupby('sub_area').id.count()[d.sub_area].fillna(0).values
    d['sub_area_volume_occupy']  = d[~d.is_investment].groupby('sub_area').id.count()[d.sub_area].fillna(0).values
    #
    d['raion_build_count_with_material_info_pc'] = d.raion_build_count_with_material_info / d.raion_popul
    d['raion_build_count_with_builddate_info_pc'] = d.raion_build_count_with_builddate_info / d.raion_popul
    #
    col_yesno  = ['culture_objects_top_25','thermal_power_plant_raion','incineration_raion','oil_chemistry_raion','radiation_raion','railroad_terminal_raion','big_market_raion','nuclear_reactor_raion','detention_facility_raion','water_1line','big_road1_1line','railroad_1line']
    for col in col_yesno:  d[col]  = (d[col]=='yes').astype(int)
    #
    col_cat    = ['month','material','state','sub_area_thd','ID_metro_thd','ID_railroad_station_walk_thd','ID_railroad_station_avto_thd','ID_big_road1_thd','ID_big_road2_thd','ID_railroad_terminal_thd','ID_bus_terminal_thd','ecology']
    d  = pd.concat([d,pd.get_dummies(d[col_cat],columns=col_cat,prefix=col_cat,prefix_sep='=')],1)
    #
    return d
