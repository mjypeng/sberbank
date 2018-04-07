from xgb_common import *

tr  = pd.read_csv('train.csv')
tt  = pd.read_csv('test.csv')
tt['material']  = tt.material.astype(float)

macro = pd.read_csv('macro.csv')
macro_col  = ['cpi','ppi','balance_trade','balance_trade_growth','usdrub','eurrub','average_provision_of_build_contract','average_provision_of_build_contract_moscow','rts','micex','micex_rgbi_tr','micex_cbi_tr','deposits_value','deposits_growth','deposits_rate','mortgage_value','mortgage_growth','mortgage_rate','income_per_cap','real_dispos_income_per_cap_growth','fixed_basket','rent_price_4+room_bus','rent_price_3room_bus','rent_price_2room_bus','rent_price_1room_bus','rent_price_3room_eco','rent_price_2room_eco','rent_price_1room_eco','theaters_viewers_per_1000_cap','seats_theather_rfmin_per_100000_cap','museum_visitis_per_100_cap','apartment_build','apartment_fund_sqm']
tr  = tr.merge(macro[['timestamp']+macro_col],how='left',on='timestamp',copy=False)
tt  = tt.merge(macro[['timestamp']+macro_col],how='left',on='timestamp',copy=False)

clean_data(tr)
clean_data(tt)

tr,col  = get_features(tr)
dtr     = dataframe_to_dmatrix(tr,col)

tt,col  = get_features(tt)
dtt     = dataframe_to_dmatrix(tt,col)

exit(0)

#-- 1st Round optimize 'max_depth','min_child_weight','gamma' --#
validation_partition  = '2014-11-06' #'2015-04-01' #'2014-10-01' #'2015-06-01' #'2014-07-01' #'2015-01-01' # 'cv' #
expt  = pd.DataFrame(
    columns=('tr1','tr1_std','vn','vn_std','vnopt','optn','tr','avg log pred','prjtt'),
    index=pd.MultiIndex.from_product(
        [[validation_partition],range(2,7),range(1,8,2),[0,0.2,0.4],[1],[0.6],[1],[4],[0.1],[20]],
        names=('vn_part','max_depth','min_child_weight','gamma','subsample','colsample_bytree','lambda','alpha','eta','early_stop')))
best_vn       = np.inf
for idx in expt.index:
    t0  = time.clock()
    xgb_params  = idx_to_xgb_params(idx,expt.index.names)
    #
    if validation_partition == 'cv':
        opt_nboost,cv_evals = xgb_cross_validation(xgb_params,tr,col)
        expt.loc[idx,['tr1','tr1_std','vn','vn_std','optn']]  = [cv_evals.tr1.iloc[-1],cv_evals.tr1_std.iloc[-1],cv_evals.vn.iloc[-1],cv_evals.vn_std.iloc[-1],opt_nboost]
    else:
        opt_nboost,tv_evals = xgb_temporal_validation(xgb_params,tr,col)
        expt.loc[idx,['tr1','vn','optn']]  = [tv_evals.tr1.iloc[-1],tv_evals.vn.iloc[-1],opt_nboost]
    #
    model  = xgb.train(xgb_params,dtr,num_boost_round=opt_nboost)
    price_tr_hat  = np.exp(model.predict(dtr)) - 1
    price_tt_hat  = np.exp(model.predict(dtt)) - 1
    expt.loc[idx,['tr','avg log pred']]  = [np.sqrt(np.mean((np.log1p(tr.price_doc)-np.log1p(price_tr_hat))**2)),np.mean(np.log1p(price_tt_hat))]
    #
    if best_vn > expt.loc[idx,'vn']:
        best_vn   = expt.loc[idx,'vn']
        feat_rank = xgb_feat_rank(model,col)
    #
    print idx
    print expt.loc[[idx]].reset_index(drop=True)
    print (time.clock()-t0)

expt['vnopt']  = expt.vn - expt.tr1
expt['prjtt']  = expt.tr + expt.vnopt
if validation_partition == '2015-01-01':
    expt['modeled public tt']  = expt['tr'] + a_20150101 + b_20150101*expt['vnopt']
elif validation_partition == '2014-07-01':
    expt['modeled public tt']  = expt['tr'] + a_20140701 + b_20140701*expt['vnopt']

expt.to_pickle('xgb_optparam_20170506_' + validation_partition.replace('-','') + '_1st.pkl')
feat_rank.to_pickle('xgb_optparam_20170506_' + validation_partition.replace('-','') + '_1st_feat_rank.pkl')

feat_rank.to_clipboard('\t')
print expt.sort_values('vn')
expt.to_clipboard('\t')

exit(0)

# expt  = pd.read_pickle('xgb_optparam_20170505_20150601_1st.pkl')

expt  = pd.concat([pd.read_pickle("xgb_optparam_20170505_%s_%s.pkl" % (dt,ver)).drop('modeled public tt','columns',errors='ignore') for dt in ['20140701','20141001','20150101','20150401','20150601'] for ver in ['1st','1st-2']],0).astype(float) #
expt  = expt.groupby(level=range(1,10)).mean()
expt.to_clipboard('\t')
expt['optn']  = np.round(expt.optn).astype(int)
submit_expt(expt.sort_values('vn')[:10],tr,tt,col,"20170505_tv_all")
submit_expt(expt.sort_values('prjtt')[:10],tr,tt,col,"20170505_tv_all-2")


# expt  = pd.read_pickle('xgb_optparam_20170504_20150101_1st-2.pkl')
# submit_expt(expt.sort_values('vn')[:4],tr,tt,col,"20170504_20150101_vn")
# submit_expt(expt.sort_values('modeled public tt')[:4],tr,tt,col,"20170504_20150101_tt")

#-- 2nd Round Optimize 'subsample','colsample_bytree' --#
optidx_1st  = expt.sort_values('vn').index[:10].tolist()
expt        = pd.DataFrame(
    columns=('tr1','tr1_std','vn','vn_std','vnopt','optn','tr','avg log pred','prjtt'),
    index=pd.MultiIndex.from_tuples(
        [x[:4]+(samp,colsamp)+x[6:]
            for x in optidx_1st
            for samp in [0.5,0.6,0.7,0.8,0.9,1.0]
            for colsamp in [0.5,0.6,0.7,0.8,0.9,1.0]],
        names=expt.index.names))
best_vn       = np.inf
for idx in expt.index:
    t0  = time.clock()
    xgb_params  = idx_to_xgb_params(idx,expt.index.names)
    #
    if validation_partition == 'cv':
        opt_nboost,cv_evals = xgb_cross_validation(xgb_params,tr,col)
        expt.loc[idx,['tr1','tr1_std','vn','vn_std','optn']]  = [cv_evals.tr1.iloc[-1],cv_evals.tr1_std.iloc[-1],cv_evals.vn.iloc[-1],cv_evals.vn_std.iloc[-1],opt_nboost]
    else:
        opt_nboost,tv_evals = xgb_temporal_validation(xgb_params,tr,col)
        expt.loc[idx,['tr1','vn','optn']]  = [tv_evals.tr1.iloc[-1],tv_evals.vn.iloc[-1],opt_nboost]
    #
    model  = xgb.train(xgb_params,dtr,num_boost_round=opt_nboost)
    price_tr_hat  = np.exp(model.predict(dtr)) - 1
    price_tt_hat  = np.exp(model.predict(dtt)) - 1
    expt.loc[idx,['tr','avg log pred']]  = [np.sqrt(np.mean((np.log1p(tr.price_doc)-np.log1p(price_tr_hat))**2)),np.mean(np.log1p(price_tt_hat))]
    #
    if best_vn > expt.loc[idx,'vn']:
        best_vn   = expt.loc[idx,'vn']
        feat_rank = xgb_feat_rank(model,col)
    #
    print idx
    print expt.loc[[idx]].reset_index(drop=True)
    print (time.clock()-t0)

expt['vnopt']  = expt.vn - expt.tr1
expt['prjtt']  = expt.tr + expt.vnopt
if validation_partition == '2015-01-01':
    expt['modeled public tt']  = expt['tr'] + a_20150101 + b_20150101*expt['vnopt']

expt.to_pickle('xgb_optparam_20170506_' + validation_partition.replace('-','') + '_2nd.pkl')
feat_rank.to_pickle('xgb_optparam_20170506_' + validation_partition.replace('-','') + '_2nd_feat_rank.pkl')

feat_rank.to_clipboard('\t')
print expt.sort_values('vn')
expt.to_clipboard('\t')

exit(0)

expt  = pd.read_pickle('xgb_optparam_20170506_' + validation_partition.replace('-','') + '_2nd.pkl')
submit_expt(expt.sort_values('vn')[:10],tr,tt,col,"20170506_%s" % validation_partition.replace('-',''))


#-- 3rd Round Optimize 'lambda' and 'alpha' --#
optidx_2nd  = expt.sort_values('vn').index[:10].tolist()
expt        = pd.DataFrame(
    columns=('tr1','vn','vnopt','optn','tr','prjtt'),
    index=pd.MultiIndex.from_tuples(
        [x[:6]+(reg_lambda,reg_alpha)+x[8:]
            for x in optidx_1st
            for reg_lambda in 2.0**np.arange(-5,5)
            for reg_alpha in 2.0**np.arange(-5,5)],
        names=expt.index.names))
for idx in expt.index:
    t0  = time.clock()
    xgb_params  = idx_to_xgb_params(idx,expt.index.names)
    #
    opt_nboost,tv_evals = xgb_temporal_validation(xgb_params,tr,col)
    #
    model  = xgb.train(xgb_params,dtr,num_boost_round=opt_nboost)
    price_tr_hat  = np.exp(model.predict(dtr)) - 1
    price_tt_hat  = np.exp(model.predict(dtt)) - 1
    tt['price_doc']  = price_tt_hat
    expt.loc[idx,['tr1','vn','optn','tr']]  = [tv_evals['tr1']['rmse'][opt_nboost-1],tv_evals['vn']['rmse'][opt_nboost-1],opt_nboost,np.sqrt(np.mean((np.log(tr.price_doc+1)-np.log(price_tr_hat+1))**2))]
    #
    print idx
    print expt.loc[[idx]].reset_index(drop=True)
    print (time.clock()-t0)

expt['vnopt']  = expt.vn - expt.tr1
expt['prjtt']  = expt.tr + expt.vnopt
expt.to_pickle('xgb_optparam_20170502_3rd.pkl')

print expt.sort_values('vn')
expt.to_clipboard('\t')



expt  = pd.read_pickle('xgb_optparam_20170502_1st.pkl').sort_values('vn')[:10]
best_vn  = np.inf
for idx in expt.index:
    t0  = time.clock()
    xgb_params  = idx_to_xgb_params(idx,expt.index.names)
    #
    ID_prices  = get_ID_median_prices(tr[tr.timestamp<xgb_params['vn_part_date']])
    tr,col     = get_features(tr,ID_prices)
    #
    opt_nboost,tv_evals = xgb_temporal_validation(xgb_params,tr,col)
    expt.loc[idx,['tr1','vn','optn']]  = [tv_evals['tr1']['rmse'][opt_nboost-1],tv_evals['vn']['rmse'][opt_nboost-1],opt_nboost]
    #
    if best_vn > expt.loc[idx,'vn']:
        best_vn    = expt.loc[idx,'vn']
        model      = xgb.train(xgb_params,dataframe_to_dmatrix(tr[tr.timestamp<xgb_params['vn_part_date']],col),num_boost_round=opt_nboost)
        feat_rank  = xgb_feat_rank(model,col)
    #
    ID_prices  = get_ID_median_prices(tr)
    tr,col     = get_features(tr,ID_prices)
    dtr        = dataframe_to_dmatrix(tr,col)
    model         = xgb.train(xgb_params,dtr,num_boost_round=opt_nboost)
    price_tr_hat  = np.exp(model.predict(dtr)) - 1
    expt.loc[idx,'tr']  = np.sqrt(np.mean((np.log(tr.price_doc+1)-np.log(price_tr_hat+1))**2))
    #
    print idx
    print expt.loc[[idx]].reset_index(drop=True)
    print (time.clock()-t0)

expt['vnopt']  = expt.vn - expt.tr1
expt['prjtt']  = expt.tr + expt.vnopt
expt.to_clipboard('\t')
feat_rank.to_clipboard('\t')




#-- Final Round Parameter Tuning --#
optidx_3rd  = expt_results3.sort_values('vn').index[:10].tolist()
expt_results4  = pd.DataFrame(
    columns=('tr1','vn','optn','tr'),
    index=pd.MultiIndex.from_tuples([x[:8]+(eta,)+x[9:] for x in optidx_3rd for eta in [0.05,0.1,0.2,0.3,0.4]],
    names=expt_results3.index.names[:8]+('eta',)+expt_results3.index.names[9:]))
for idx in expt_results4.index:
    t0  = time.clock()
    vn_part_date,depth,child,gamma,subsamp,colsamp,reg_lambda,reg_alpha,eta,early_stop  = idx
    xgb_params = {'objective':'reg:linear',
                  'eval_metric':'rmse',
                  'silent':1,
                  'booster':'gbtree',
                    'eta':eta,
                    'max_depth':depth,
                    'min_child_weight':child,
                    'subsample':subsamp,
                    'colsample_bytree':colsamp,
                    'lambda':reg_lambda,
                    'gamma':gamma,
                    'alpha':reg_alpha}
    dtr  = dataframe_to_dmatrix(tr[tr.timestamp<vn_part_date],col)
    dvn  = dataframe_to_dmatrix(tr[tr.timestamp>=vn_part_date],col)
    tv_evals = {}
    tv_model = xgb.train(xgb_params,dtr,num_boost_round=1000,evals=[(dtr,'tr1'),(dvn,'vn')],early_stopping_rounds=early_stop,evals_result=tv_evals,verbose_eval=False)
    opt_n_booster = tv_model.best_ntree_limit
    #
    dtr  = dataframe_to_dmatrix(tr,col)
    dtt  = dataframe_to_dmatrix(tt,col)
    tr_evals = {}
    model    = xgb.train(xgb_params,dtr,num_boost_round=opt_n_booster)
    price_tr_hat  = np.exp(model.predict(dtr)) - 1
    price_tt_hat  = np.exp(model.predict(dtt)) - 1
    tt['price_doc']  = price_tt_hat
    expt_results4.loc[idx,['tr1','vn','optn','tr']]  = [
        tv_evals['tr1']['rmse'][opt_n_booster-1],
        tv_evals['vn']['rmse'][opt_n_booster-1],
        opt_n_booster,
        np.sqrt(np.mean((np.log(tr.price_doc+1)-np.log(price_tr_hat+1))**2))]
    print expt_results4.loc[[idx]],(time.clock()-t0)

expt_results4['vnopt']  = expt_results4.vn - expt_results4.tr1
expt_results4['prjtt']  = expt_results4.tr + expt_results4.vnopt

print expt_results4.sort_values('vn')

expt_results4.to_clipboard('\t')
expt_results4.to_pickle('xgb_optparam_temp.pkl')
# expt_results4  = pd.read_pickle('xgb_optparam_20170501-2_4th.pkl')

#-- "Submit" top 10 results --#
expt_results_final  = expt_results4.sort_values('vn')[:10]
for i,idx in enumerate(expt_results_final.index):
    t0  = time.clock()
    vn_part_date,depth,child,gamma,subsamp,colsamp,reg_lambda,reg_alpha,eta,early_stop  = idx
    xgb_params = {'objective':'reg:linear',
                  'eval_metric':'rmse',
                  'silent':1,
                  'booster':'gbtree',
                    'eta':eta,
                    'max_depth':depth,
                    'min_child_weight':child,
                    'subsample':subsamp,
                    'colsample_bytree':colsamp,
                    'lambda':reg_lambda,
                    'gamma':gamma,
                    'alpha':reg_alpha}
    opt_n_booster = expt_results_final.loc[idx,'optn']
    #
    dtr  = dataframe_to_dmatrix(tr,col)
    dtt  = dataframe_to_dmatrix(tt,col)
    model    = xgb.train(xgb_params,dtr,num_boost_round=opt_n_booster)
    price_tr_hat  = np.exp(model.predict(dtr)) - 1
    price_tt_hat  = np.exp(model.predict(dtt)) - 1
    if np.abs(expt_results_final.loc[idx,'tr'] - np.sqrt(np.mean((np.log(tr.price_doc+1)-np.log(price_tr_hat+1))**2))) < 10**-7: print "OK"
    else: print "Warning: metric mismatch"
    tt['price_doc']  = price_tt_hat
    submit(tt,"20170502_%d_%s" % (i+1,str(idx)))

expt_results_final.to_clipboard('\t')
