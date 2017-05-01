import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import xgboost as xgb

sub_area_names_80  = ['Poselenie Vnukovskoe','Poselenie Sosenskoe','Solncevo','Nagatinskij Zaton','Poselenie Moskovskij','Tverskoe','Mitino','Poselenie Desjonovskoe','Zapadnoe Degunino','Mar\'ino','Juzhnoe Butovo','Nekrasovka','Poselenie Shherbinka','Otradnoe','Poselenie Novofedorovskoe','Sviblovo','Golovinskoe','Filevskij Park','Krjukovo']
sub_area_names  = ['Poselenie Vnukovskoe','Poselenie Sosenskoe','Solncevo','Nagatinskij Zaton','Poselenie Moskovskij','Tverskoe','Mitino','Poselenie Desjonovskoe','Zapadnoe Degunino','Mar\'ino','Juzhnoe Butovo','Nekrasovka','Poselenie Shherbinka','Otradnoe','Poselenie Novofedorovskoe','Sviblovo','Golovinskoe','Filevskij Park','Krjukovo']#,'Danilovskoe','Chertanovo Juzhnoe','Ljublino','Nagornoe','Ochakovo-Matveevskoe','Horoshevo-Mnevniki','Jasenevo','Gol\'janovo']#,'Vyhino-Zhulebino','Timirjazevskoe','Kuncevo','Severnoe','Chertanovo Severnoe','Kuz\'minki','Birjulevo Vostochnoe','Pokrovskoe Streshnevo']

def dataframe_to_dmatrix(d,col):
    return xgb.DMatrix(d[col].fillna(-1),label=np.log(d.price_doc + 1) if 'price_doc' in d else None)

def submit(tt,name):
    tt[['id','price_doc']].to_csv("submission_%s.csv" % name,index=False)

tr  = pd.read_csv('train.csv')
tt  = pd.read_csv('test.csv')

tr['price_per_sq']  = tr.price_doc/tr.full_sq
tt['material']  = tt.material.astype(float)

def preprocess(d):
    d['dt']  = pd.to_datetime(d.timestamp)
    d['ym']  = d.timestamp.str[:7]
    d['year']          = pd.to_numeric(d.ym.str[:4])
    d['month']         = pd.to_numeric(d.ym.str[5:])
    d['is_investment'] = d.product_type=='Investment'
    d['monthly_volume']  = d.groupby('ym').id.count()[d.ym].values
    d['monthly_volume_invest']  = d[d.is_investment].groupby('ym').id.count()[d.ym].fillna(0).values
    d['monthly_volume_occupy']  = d[~d.is_investment].groupby('ym').id.count()[d.ym].fillna(0).values
    d['build_age']     = np.where((d.build_year>1000)&(d.build_year<3000),d.year-d.build_year,np.nan)
    d['life_ratio']    = d.life_sq / d.full_sq
    d['kitch_ratio']   = d.kitch_sq / d.full_sq
    d['floor_ratio']   = d.floor / d.max_floor
    d  = pd.concat([d,
        pd.get_dummies(d.material,prefix='material',prefix_sep='='),
        pd.get_dummies(d.month,prefix='month',prefix_sep='='),
        pd.get_dummies(d.state,prefix='state',prefix_sep='='),
        pd.get_dummies(d.sub_area,prefix='sub_area',prefix_sep='=')],1)
    return d

tr  = preprocess(tr)
tt  = preprocess(tt)

tr1  = tr[tr.timestamp<'2014-07-01']
vn   = tr[tr.timestamp>='2014-07-01']

col  = ['full_sq','life_ratio','floor','max_floor','floor_ratio','material=1.0','material=2.0','material=3.0','material=5.0','material=6.0','build_age','num_room','kitch_ratio','state=2.0','state=3.0','state=4.0','is_investment','area_m','metro_km_walk','kremlin_km','shopping_centers_km','year','month','monthly_volume_invest','monthly_volume_occupy'] + ['sub_area='+name for name in sub_area_names_80] #'month=6','state=1.0','material=4.0' omitted

expt_results  = pd.DataFrame(
    columns=('tr1','vn','optn','tr'),
    index=pd.MultiIndex.from_product(
        [range(2,8),range(1,10),[0,0.1,0.2,0.3,0.4]],
        names=('max_depth','min_child_weight','gamma')))
for idx in expt_results.index:
    depth,child,gamma  = idx
    xgb_params = {'objective':'reg:linear',
                  'eval_metric':'rmse',
                  'silent':1,
                  'booster':'gbtree',
                    'eta':0.3,
                    'max_depth':depth,
                    'min_child_weight':child,
                    'subsample':0.8,
                    'colsample_bytree':1.0,#0.8,#
                    'lambda':1,
                    'gamma':gamma,
                    'alpha':0}
    dtr  = dataframe_to_dmatrix(tr1,col)
    dvn  = dataframe_to_dmatrix(vn,col)
    tv_evals = {}
    tv_model = xgb.train(xgb_params,dtr,num_boost_round=10000,evals=[(dtr,'tr1'),(dvn,'vn')],early_stopping_rounds=10,evals_result=tv_evals,verbose_eval=False)
    opt_n_booster = tv_model.best_ntree_limit
    #
    dtr  = dataframe_to_dmatrix(tr,col)
    dtt  = dataframe_to_dmatrix(tt,col)
    tr_evals = {}
    model    = xgb.train(xgb_params,dtr,num_boost_round=opt_n_booster,evals=[(dtr,'tr')],evals_result=tr_evals,verbose_eval=False)
    price_tr_hat  = np.exp(model.predict(dtr)) - 1
    price_tt_hat  = np.exp(model.predict(dtt)) - 1
    tt['price_doc']  = price_tt_hat
    expt_results.loc[idx,['tr1','vn','optn','tr']]  = [
        tv_evals['tr1']['rmse'][opt_n_booster-1],
        tv_evals['vn']['rmse'][opt_n_booster-1],
        opt_n_booster,
        np.sqrt(np.mean((np.log(tr.price_doc+1)-np.log(price_tr_hat+1))**2))]
    print expt_results.loc[[idx]]

print expt_results.sort_values('vn')

#-- 2nd Round Parameter Tuning --#
optidx_1st  = expt_results.sort_values('vn').index[:10].tolist()
expt_results2  = pd.DataFrame(
    columns=('tr1','vn','optn','tr'),
    index=pd.MultiIndex.from_tuples([x+(samp,colsamp) for x in optidx_1st for samp in [0.5,0.6,0.7,0.8,0.9,1.0] for colsamp in [0.5,0.6,0.7,0.8,0.9,1.0]],names=expt_results.index.names+('subsample','colsample_bytree')))
for idx in expt_results2.index:
    depth,child,gamma,samp,colsamp = idx
    xgb_params = {'objective':'reg:linear',
                  'eval_metric':'rmse',
                  'silent':1,
                  'booster':'gbtree',
                    'eta':0.3,
                    'max_depth':depth,
                    'min_child_weight':child,
                    'subsample':samp,
                    'colsample_bytree':colsamp,
                    'lambda':1,
                    'gamma':gamma,
                    'alpha':0}
    dtr  = dataframe_to_dmatrix(tr1,col)
    dvn  = dataframe_to_dmatrix(vn,col)
    tv_evals = {}
    tv_model = xgb.train(xgb_params,dtr,num_boost_round=10000,evals=[(dtr,'tr1'),(dvn,'vn')],early_stopping_rounds=10,evals_result=tv_evals,verbose_eval=False)
    opt_n_booster = tv_model.best_ntree_limit
    #
    dtr  = dataframe_to_dmatrix(tr,col)
    dtt  = dataframe_to_dmatrix(tt,col)
    tr_evals = {}
    model    = xgb.train(xgb_params,dtr,num_boost_round=opt_n_booster,evals=[(dtr,'tr')],evals_result=tr_evals,verbose_eval=False)
    price_tr_hat  = np.exp(model.predict(dtr)) - 1
    price_tt_hat  = np.exp(model.predict(dtt)) - 1
    tt['price_doc']  = price_tt_hat
    expt_results2.loc[idx,['tr1','vn','optn','tr']]  = [
        tv_evals['tr1']['rmse'][opt_n_booster-1],
        tv_evals['vn']['rmse'][opt_n_booster-1],
        opt_n_booster,
        np.sqrt(np.mean((np.log(tr.price_doc+1)-np.log(price_tr_hat+1))**2))]
    print expt_results2.loc[[idx]]

print expt_results2.sort_values('vn')

#-- 3rd Round Parameter Tuning --#
optidx_2nd  = expt_results2.sort_values('vn').index[:10].tolist()
expt_results3  = pd.DataFrame(
    columns=('tr1','vn','optn','tr'),
    index=pd.MultiIndex.from_tuples([x+(alpha,) for x in optidx_2nd for alpha in 2.0**np.arange(-5,5)],names=expt_results2.index.names+('alpha',)))
for idx in expt_results3.index:
    depth,child,gamma,samp,colsamp,alpha = idx
    xgb_params = {'objective':'reg:linear',
                  'eval_metric':'rmse',
                  'silent':1,
                  'booster':'gbtree',
                    'eta':0.3,
                    'max_depth':depth,
                    'min_child_weight':child,
                    'subsample':samp,
                    'colsample_bytree':colsamp,
                    'lambda':1,
                    'gamma':gamma,
                    'alpha':alpha}
    dtr  = dataframe_to_dmatrix(tr1,col)
    dvn  = dataframe_to_dmatrix(vn,col)
    tv_evals = {}
    tv_model = xgb.train(xgb_params,dtr,num_boost_round=10000,evals=[(dtr,'tr1'),(dvn,'vn')],early_stopping_rounds=10,evals_result=tv_evals,verbose_eval=False)
    opt_n_booster = tv_model.best_ntree_limit
    #
    dtr  = dataframe_to_dmatrix(tr,col)
    dtt  = dataframe_to_dmatrix(tt,col)
    tr_evals = {}
    model    = xgb.train(xgb_params,dtr,num_boost_round=opt_n_booster,evals=[(dtr,'tr')],evals_result=tr_evals,verbose_eval=False)
    price_tr_hat  = np.exp(model.predict(dtr)) - 1
    price_tt_hat  = np.exp(model.predict(dtt)) - 1
    tt['price_doc']  = price_tt_hat
    expt_results3.loc[idx,['tr1','vn','optn','tr']]  = [
        tv_evals['tr1']['rmse'][opt_n_booster-1],
        tv_evals['vn']['rmse'][opt_n_booster-1],
        opt_n_booster,
        np.sqrt(np.mean((np.log(tr.price_doc+1)-np.log(price_tr_hat+1))**2))]
    print expt_results3.loc[[idx]]

print expt_results3.sort_values('vn')

#-- Final Round Parameter Tuning --#
optidx_3rd  = expt_results3.sort_values('vn').index[:10].tolist()
expt_results4  = pd.DataFrame(
    columns=('tr1','vn','optn','tr'),
    index=pd.MultiIndex.from_tuples([x+(eta,) for x in optidx_3rd for eta in [0.05,0.1,0.2,0.3,0.4]],names=expt_results3.index.names+('eta',)))
for idx in expt_results4.index:
    depth,child,gamma,samp,colsamp,alpha,eta = idx
    xgb_params = {'objective':'reg:linear',
                  'eval_metric':'rmse',
                  'silent':1,
                  'booster':'gbtree',
                    'eta':eta,
                    'max_depth':depth,
                    'min_child_weight':child,
                    'subsample':samp,
                    'colsample_bytree':colsamp,
                    'lambda':1,
                    'gamma':gamma,
                    'alpha':alpha}
    dtr  = dataframe_to_dmatrix(tr1,col)
    dvn  = dataframe_to_dmatrix(vn,col)
    tv_evals = {}
    tv_model = xgb.train(xgb_params,dtr,num_boost_round=10000,evals=[(dtr,'tr1'),(dvn,'vn')],early_stopping_rounds=10,evals_result=tv_evals,verbose_eval=False)
    opt_n_booster = tv_model.best_ntree_limit
    #
    dtr  = dataframe_to_dmatrix(tr,col)
    dtt  = dataframe_to_dmatrix(tt,col)
    tr_evals = {}
    model    = xgb.train(xgb_params,dtr,num_boost_round=opt_n_booster,evals=[(dtr,'tr')],evals_result=tr_evals,verbose_eval=False)
    price_tr_hat  = np.exp(model.predict(dtr)) - 1
    price_tt_hat  = np.exp(model.predict(dtt)) - 1
    tt['price_doc']  = price_tt_hat
    expt_results4.loc[idx,['tr1','vn','optn','tr']]  = [
        tv_evals['tr1']['rmse'][opt_n_booster-1],
        tv_evals['vn']['rmse'][opt_n_booster-1],
        opt_n_booster,
        np.sqrt(np.mean((np.log(tr.price_doc+1)-np.log(price_tr_hat+1))**2))]
    print expt_results4.loc[[idx]]

print expt_results4.sort_values('vn')

#-- "Submit" top 10 results --#
optidx_4th  = expt_results4.sort_values('vn').index[:10].tolist()
expt_results_final  = pd.DataFrame(
    columns=('tr1','vn','optn','tr'),
    index=pd.MultiIndex.from_tuples(optidx_4th,names=expt_results4.index.names))
for i,idx in enumerate(expt_results_final.index):
    depth,child,gamma,samp,colsamp,alpha,eta = idx
    xgb_params = {'objective':'reg:linear',
                  'eval_metric':'rmse',
                  'silent':1,
                  'booster':'gbtree',
                    'eta':eta,
                    'max_depth':depth,
                    'min_child_weight':child,
                    'subsample':samp,
                    'colsample_bytree':colsamp,
                    'lambda':1,
                    'gamma':gamma,
                    'alpha':alpha}
    dtr  = dataframe_to_dmatrix(tr1,col)
    dvn  = dataframe_to_dmatrix(vn,col)
    tv_evals = {}
    tv_model = xgb.train(xgb_params,dtr,num_boost_round=10000,evals=[(dtr,'tr1'),(dvn,'vn')],early_stopping_rounds=10,evals_result=tv_evals,verbose_eval=False)
    opt_n_booster = tv_model.best_ntree_limit
    #
    dtr  = dataframe_to_dmatrix(tr,col)
    dtt  = dataframe_to_dmatrix(tt,col)
    tr_evals = {}
    model    = xgb.train(xgb_params,dtr,num_boost_round=opt_n_booster,evals=[(dtr,'tr')],evals_result=tr_evals,verbose_eval=False)
    price_tr_hat  = np.exp(model.predict(dtr)) - 1
    price_tt_hat  = np.exp(model.predict(dtt)) - 1
    tt['price_doc']  = price_tt_hat
    submit(tt,"20170430_%d_%s" % (i+1,str(idx)))
    expt_results_final.loc[idx,['tr1','vn','optn','tr']]  = [
        tv_evals['tr1']['rmse'][opt_n_booster-1],
        tv_evals['vn']['rmse'][opt_n_booster-1],
        opt_n_booster,
        np.sqrt(np.mean((np.log(tr.price_doc+1)-np.log(price_tr_hat+1))**2))]
    print expt_results_final.loc[[idx]]

print expt_results_final.sort_values('vn')
