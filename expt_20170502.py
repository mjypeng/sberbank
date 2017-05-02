from common import *

def pearsonr_w_price(d,col):
    mask  = d[col].notnull()
    return stats.pearsonr(np.log(d[mask].price_doc+1),d.loc[mask,col])

def get_feat_rank(model,dtr,tr=None):
    feat_rank  = pd.concat([
        pd.Series(model.get_score(importance_type='weight')),
        pd.Series(model.get_score(importance_type='gain')),
        pd.Series(model.get_score(importance_type='cover'))],
        1,keys=('wt','gain','cover')).reindex(index=dtr.feature_names)
    if tr is not None:
        for col in dtr.feature_names:
            r,p  = pearsonr_w_price(tr,col)
            feat_rank.loc[col,'corr_w_price'] = r
            feat_rank.loc[col,'corr_w_price_p'] = p
    return feat_rank

def dataframe_to_dmatrix(d,col):
    return xgb.DMatrix(d[col].fillna(-1),label=np.log(d.price_doc + 1) if 'price_doc' in d else None)

tr  = pd.read_csv('train.csv')
tt  = pd.read_csv('test.csv')
tt['material']  = tt.material.astype(float)

macro = pd.read_csv('macro.csv')
tr  = tr.merge(macro[['timestamp','cpi','ppi']],how='left',on='timestamp',copy=False)
tt  = tt.merge(macro[['timestamp','cpi','ppi']],how='left',on='timestamp',copy=False)

col_cat_thd  = {'sub_area':80, 'ID_metro':99, 'ID_railroad_station_walk':200, 'ID_railroad_station_avto':200, 'ID_big_road1':300, 'ID_big_road2':300, 'ID_railroad_terminal':300, 'ID_bus_terminal':500}
for col in col_cat_thd:
    tr[col+'_thd'] = np.where(tr[col].isin(get_dummy_values(tt,col,col_cat_thd[col])),tr[col],np.nan)
    tt[col+'_thd'] = np.where(tt[col].isin(get_dummy_values(tt,col,col_cat_thd[col])),tt[col],np.nan)

tr['price_per_sq']  = tr.price_doc/tr.full_sq

tr  = preprocess(tr)
tt  = preprocess(tt)

col_sets  = [
    ['year','month','monthly_volume_invest','monthly_volume_occupy'],
    ['full_sq','life_ratio','floor','max_floor','floor_ratio'],
    ['material=1.0','material=2.0','material=3.0','material=4.0','material=5.0','material=6.0'],
    ['build_age','num_room','kitch_ratio'],
    ['state=1.0','state=2.0','state=3.0','state=4.0'],
    ['is_investment'],
    ['sub_area_thd='+name for name in get_dummy_values(tt,'sub_area',80)],
    ['area_m','raion_popul','sub_area_pop_density'],#,'sub_area_volume','sub_area_volume_invest','sub_area_volume_occupy'],
    ['green_zone_part','indust_part'],
    ['children_preschool','preschool_quota','preschool_education_centers_raion','children_school','school_quota','school_education_centers_raion','school_education_centers_top_20_raion'],
    ['hospital_beds_raion','healthcare_centers_raion','university_top_20_raion','sport_objects_raion','additional_education_raion'],
    ['culture_objects_top_25','culture_objects_top_25_raion','shopping_centers_raion','office_raion','thermal_power_plant_raion','incineration_raion','oil_chemistry_raion','radiation_raion','railroad_terminal_raion','big_market_raion','nuclear_reactor_raion','detention_facility_raion'],
    ['full_all','male_f','female_f','young_all','young_male','young_female','work_all','work_male','work_female','ekder_all','ekder_male','ekder_female'],
    ['0_6_all','0_6_male','0_6_female','7_14_all','7_14_male','7_14_female','0_17_all','0_17_male','0_17_female','16_29_all','16_29_male','16_29_female','0_13_all','0_13_male','0_13_female'],
    ['raion_build_count_with_material_info','raion_build_count_with_material_info_pc','build_count_block','build_count_wood','build_count_frame','build_count_brick','build_count_monolith','build_count_panel','build_count_foam','build_count_slag','build_count_mix'],
    ['raion_build_count_with_builddate_info','raion_build_count_with_builddate_info_pc','build_count_before_1920','build_count_1921-1945','build_count_1946-1970','build_count_1971-1995','build_count_after_1995'],
    ["ID_metro_thd=%.1f" % name for name in get_dummy_values(tt,'ID_metro',99)],
    ['metro_min_avto','metro_km_avto','metro_min_walk','metro_km_walk'],
    ['kindergarten_km','school_km','park_km','green_zone_km','industrial_km','water_treatment_km','cemetery_km','incineration_km'],
    ['railroad_station_walk_km','railroad_station_walk_min','railroad_station_avto_km','railroad_station_avto_min','public_transport_station_km','public_transport_station_min_walk'],
    ["ID_railroad_station_walk_thd=%.1f" % name for name in get_dummy_values(tt,'ID_railroad_station_walk',200)],
    ["ID_railroad_station_avto_thd=%.1f" % name for name in get_dummy_values(tt,'ID_railroad_station_avto',200)],
    ['water_km','water_1line'],
    ['mkad_km','ttk_km','sadovoe_km','bulvar_ring_km','kremlin_km'],
    ['big_road1_km','big_road1_1line','big_road2_km','railroad_km','railroad_1line'],
    ["ID_big_road1_thd=%.1f" % name for name in get_dummy_values(tt,'ID_big_road1',300)],
    ["ID_big_road2_thd=%.1f" % name for name in get_dummy_values(tt,'ID_big_road2',300)],
    ['zd_vokzaly_avto_km'],
    ["ID_railroad_terminal_thd=%.1f" % name for name in get_dummy_values(tt,'ID_railroad_terminal',300)],
    ['bus_terminal_avto_km'],
    ["ID_bus_terminal_thd=%.1f" % name for name in get_dummy_values(tt,'ID_bus_terminal',500)],
    ['oil_chemistry_km','nuclear_reactor_km','radiation_km','power_transmission_line_km','thermal_power_plant_km','ts_km'],
    ['big_market_km','market_shop_km','fitness_km','swim_pool_km','ice_rink_km','stadium_km','basketball_km'],
    ['hospice_morgue_km','detention_facility_km','public_healthcare_km'],
    ['university_km','workplaces_km','shopping_centers_km','office_km','additional_education_km','preschool_km','big_church_km','church_synagogue_km','mosque_km','theater_km','museum_km','exhibition_km','catering_km'],
    ['ecology='+name for name in tt.ecology.unique()],
    ['green_part_500','prom_part_500','office_count_500','office_sqm_500','trc_count_500','trc_sqm_500','cafe_count_500','cafe_sum_500_min_price_avg','cafe_sum_500_max_price_avg','cafe_avg_price_500','cafe_count_500_na_price','cafe_count_500_price_500','cafe_count_500_price_1000','cafe_count_500_price_1500','cafe_count_500_price_2500','cafe_count_500_price_4000','cafe_count_500_price_high','big_church_count_500','church_count_500','mosque_count_500','leisure_count_500','sport_count_500','market_count_500'],
    ['green_part_1000','prom_part_1000','office_count_1000','office_sqm_1000','trc_count_1000','trc_sqm_1000','cafe_count_1000','cafe_sum_1000_min_price_avg','cafe_sum_1000_max_price_avg','cafe_avg_price_1000','cafe_count_1000_na_price','cafe_count_1000_price_500','cafe_count_1000_price_1000','cafe_count_1000_price_1500','cafe_count_1000_price_2500','cafe_count_1000_price_4000','cafe_count_1000_price_high','big_church_count_1000','church_count_1000','mosque_count_1000','leisure_count_1000','sport_count_1000','market_count_1000'],
    ['green_part_1500','prom_part_1500','office_count_1500','office_sqm_1500','trc_count_1500','trc_sqm_1500','cafe_count_1500','cafe_sum_1500_min_price_avg','cafe_sum_1500_max_price_avg','cafe_avg_price_1500','cafe_count_1500_na_price','cafe_count_1500_price_500','cafe_count_1500_price_1000','cafe_count_1500_price_1500','cafe_count_1500_price_2500','cafe_count_1500_price_4000','cafe_count_1500_price_high','big_church_count_1500','church_count_1500','mosque_count_1500','leisure_count_1500','sport_count_1500','market_count_1500'],
    ['green_part_2000','prom_part_2000','office_count_2000','office_sqm_2000','trc_count_2000','trc_sqm_2000','cafe_count_2000','cafe_sum_2000_min_price_avg','cafe_sum_2000_max_price_avg','cafe_avg_price_2000','cafe_count_2000_na_price','cafe_count_2000_price_500','cafe_count_2000_price_1000','cafe_count_2000_price_1500','cafe_count_2000_price_2500','cafe_count_2000_price_4000','cafe_count_2000_price_high','big_church_count_2000','church_count_2000','mosque_count_2000','leisure_count_2000','sport_count_2000','market_count_2000'],
    ['green_part_3000','prom_part_3000','office_count_3000','office_sqm_3000','trc_count_3000','trc_sqm_3000','cafe_count_3000','cafe_sum_3000_min_price_avg','cafe_sum_3000_max_price_avg','cafe_avg_price_3000','cafe_count_3000_na_price','cafe_count_3000_price_500','cafe_count_3000_price_1000','cafe_count_3000_price_1500','cafe_count_3000_price_2500','cafe_count_3000_price_4000','cafe_count_3000_price_high','big_church_count_3000','church_count_3000','mosque_count_3000','leisure_count_3000','sport_count_3000','market_count_3000'],
    ['green_part_5000','prom_part_5000','office_count_5000','office_sqm_5000','trc_count_5000','trc_sqm_5000','cafe_count_5000','cafe_sum_5000_min_price_avg','cafe_sum_5000_max_price_avg','cafe_avg_price_5000','cafe_count_5000_na_price','cafe_count_5000_price_500','cafe_count_5000_price_1000','cafe_count_5000_price_1500','cafe_count_5000_price_2500','cafe_count_5000_price_4000','cafe_count_5000_price_high','big_church_count_5000','church_count_5000','mosque_count_5000','leisure_count_5000','sport_count_5000','market_count_5000']]
col  = [x for y in col_sets for x in y]

vn_partition_date  = '2014-07-01' #'2015-01-01' #
expt_results  = pd.DataFrame(
    columns=('tr1','vn','optn','tr'),
    index=pd.MultiIndex.from_product(
        [[vn_partition_date],range(2,8),range(1,8,2),[0,0.1,0.2,0.4],[0.8],[0.2],[30]],
        names=('vn_part_date','max_depth','min_child_weight','gamma','colsample_bytree','eta','early_stop')))
for idx in expt_results.index:
    t0  = time.clock()
    vn_part_date,depth,child,gamma,colsamp,eta,early_stop  = idx
    xgb_params = {'objective':'reg:linear',
                  'eval_metric':'rmse',
                  'silent':1,
                  'booster':'gbtree',
                    'eta':eta,
                    'max_depth':depth,
                    'min_child_weight':child,
                    'subsample':0.8,
                    'colsample_bytree':colsamp,
                    'lambda':1,
                    'gamma':gamma,
                    'alpha':0}
    dtr  = dataframe_to_dmatrix(tr[tr.timestamp<vn_part_date],col)
    dvn  = dataframe_to_dmatrix(tr[tr.timestamp>=vn_part_date],col)
    tv_evals = {}
    tv_model = xgb.train(xgb_params,dtr,num_boost_round=1000,evals=[(dtr,'tr1'),(dvn,'vn')],early_stopping_rounds=early_stop,evals_result=tv_evals,verbose_eval=False)
    opt_n_booster = tv_model.best_ntree_limit
    #
    dtr  = dataframe_to_dmatrix(tr,col)
    dtt  = dataframe_to_dmatrix(tt,col)
    tr_evals = {}
    model    = xgb.train(xgb_params,dtr,num_boost_round=opt_n_booster)#,evals=[(dtr,'tr')],evals_result=tr_evals,verbose_eval=False)
    price_tr_hat  = np.exp(model.predict(dtr)) - 1
    price_tt_hat  = np.exp(model.predict(dtt)) - 1
    tt['price_doc']  = price_tt_hat
    expt_results.loc[idx,['tr1','vn','optn','tr']]  = [
        tv_evals['tr1']['rmse'][opt_n_booster-1],
        tv_evals['vn']['rmse'][opt_n_booster-1],
        opt_n_booster,
        np.sqrt(np.mean((np.log(tr.price_doc+1)-np.log(price_tr_hat+1))**2))]
    print expt_results.loc[[idx]],(time.clock()-t0)

expt_results['vnopt']  = expt_results.vn - expt_results.tr1
expt_results['prjtt']  = expt_results.tr + expt_results.vnopt

print expt_results.sort_values('vn')

expt_results.to_clipboard('\t')
expt_results.to_pickle('xgb_optparam_temp.pkl')

#-- 2nd Round Parameter Tuning --#
optidx_1st  = expt_results.sort_values('vn').index[:10].tolist()
expt_results2  = pd.DataFrame(
    columns=('tr1','vn','optn','tr'),
    index=pd.MultiIndex.from_tuples(
        [x[:4]+(samp,colsamp)+x[5:]
            for x in optidx_1st
            for samp in [0.5,0.6,0.7,0.8,0.9,1.0]
            for colsamp in [0.5,0.6,0.7,0.8,0.9,1.0]],
        names=expt_results.index.names[:4]+('subsample','colsample_bytree')+expt_results.index.names[5:]))
for idx in expt_results2.index:
    t0  = time.clock()
    vn_part_date,depth,child,gamma,subsamp,colsamp,eta,early_stop  = idx
    xgb_params = {'objective':'reg:linear',
                  'eval_metric':'rmse',
                  'silent':1,
                  'booster':'gbtree',
                    'eta':eta,
                    'max_depth':depth,
                    'min_child_weight':child,
                    'subsample':subsamp,
                    'colsample_bytree':colsamp,
                    'lambda':1,
                    'gamma':gamma,
                    'alpha':0}
    dtr  = dataframe_to_dmatrix(tr[tr.timestamp<vn_part_date],col)
    dvn  = dataframe_to_dmatrix(tr[tr.timestamp>=vn_part_date],col)
    tv_evals = {}
    tv_model = xgb.train(xgb_params,dtr,num_boost_round=1000,evals=[(dtr,'tr1'),(dvn,'vn')],early_stopping_rounds=early_stop,evals_result=tv_evals,verbose_eval=False)
    opt_n_booster = tv_model.best_ntree_limit
    #
    dtr  = dataframe_to_dmatrix(tr,col)
    dtt  = dataframe_to_dmatrix(tt,col)
    tr_evals = {}
    model    = xgb.train(xgb_params,dtr,num_boost_round=opt_n_booster)#,evals=[(dtr,'tr')],evals_result=tr_evals,verbose_eval=False)
    price_tr_hat  = np.exp(model.predict(dtr)) - 1
    price_tt_hat  = np.exp(model.predict(dtt)) - 1
    tt['price_doc']  = price_tt_hat
    expt_results2.loc[idx,['tr1','vn','optn','tr']]  = [
        tv_evals['tr1']['rmse'][opt_n_booster-1],
        tv_evals['vn']['rmse'][opt_n_booster-1],
        opt_n_booster,
        np.sqrt(np.mean((np.log(tr.price_doc+1)-np.log(price_tr_hat+1))**2))]
    print expt_results2.loc[[idx]],(time.clock()-t0)

expt_results2['vnopt']  = expt_results2.vn - expt_results2.tr1
expt_results2['prjtt']  = expt_results2.tr + expt_results2.vnopt

print expt_results2.sort_values('vn')

expt_results2.to_clipboard('\t')
expt_results2.to_pickle('xgb_optparam_temp.pkl')
# expt_results2  = pd.read_pickle('xgb_optparam_20170501-2_2nd.pkl')

#-- 3rd Round Parameter Tuning --#
optidx_2nd  = expt_results2.sort_values('vn').index[:10].tolist()
expt_results3  = pd.DataFrame(
    columns=('tr1','vn','optn','tr'),
    index=pd.MultiIndex.from_tuples([x[:6]+(reg_lambda,reg_alpha)+x[6:] for x in optidx_2nd for reg_lambda in 2.0**np.arange(-5,5) for reg_alpha in 2.0**np.arange(-5,5)],
    names=expt_results2.index.names[:6]+('lambda','alpha')+expt_results2.index.names[6:]))
for idx in expt_results3.index:
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
    expt_results3.loc[idx,['tr1','vn','optn','tr']]  = [
        tv_evals['tr1']['rmse'][opt_n_booster-1],
        tv_evals['vn']['rmse'][opt_n_booster-1],
        opt_n_booster,
        np.sqrt(np.mean((np.log(tr.price_doc+1)-np.log(price_tr_hat+1))**2))]
    print expt_results3.loc[[idx]],(time.clock()-t0)

expt_results3['vnopt']  = expt_results3.vn - expt_results3.tr1
expt_results3['prjtt']  = expt_results3.tr + expt_results3.vnopt

print expt_results3.sort_values('vn')

expt_results3.to_clipboard('\t')
expt_results3.to_pickle('xgb_optparam_temp.pkl')
# expt_results3  = pd.read_pickle('xgb_optparam_20170501-2_3rd.pkl')

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

#-- Extra round --#
expt_results_extra  = pd.DataFrame(
    columns=('tr1','vn','optn','tr'),
    index=pd.MultiIndex.from_product(
        [['2014-07-01'],range(2,8),range(1,8,2),[0,0.1,0.2,0.4],[1],[0.5],[2],[8],[0.2],[50]],
        names=('vn_part_date','max_depth','min_child_weight','gamma','subsample','colsample_bytree','lambda','alpha','eta','early_stop')))
for idx in expt_results_extra.index:
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
    model    = xgb.train(xgb_params,dtr,num_boost_round=opt_n_booster)
    price_tr_hat  = np.exp(model.predict(dtr)) - 1
    price_tt_hat  = np.exp(model.predict(dtt)) - 1
    tt['price_doc']  = price_tt_hat
    expt_results_extra.loc[idx,['tr1','vn','optn','tr']]  = [
        tv_evals['tr1']['rmse'][opt_n_booster-1],
        tv_evals['vn']['rmse'][opt_n_booster-1],
        opt_n_booster,
        np.sqrt(np.mean((np.log(tr.price_doc+1)-np.log(price_tr_hat+1))**2))]
    print expt_results_extra.loc[[idx]],(time.clock()-t0)

expt_results_extra['vnopt']  = expt_results_extra.vn - expt_results_extra.tr1
expt_results_extra['prjtt']  = expt_results_extra.tr + expt_results_extra.vnopt

print expt_results_extra.sort_values('vn')

expt_results_extra.to_clipboard('\t')
expt_results_extra.to_pickle('xgb_optparam_temp.pkl')

expt_results_extra  = pd.read_pickle('xgb_optparam_20170501-2_extra.pkl')

expt_results_extra.sort_values('vn',inplace=True)
vn_part_date,depth,child,gamma,subsamp,colsamp,reg_lambda,reg_alpha,eta,early_stop  = expt_results_extra.index[0]
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
opt_n_booster = expt_results_extra.iloc[0].optn
#
dtr  = dataframe_to_dmatrix(tr,col)
dtt  = dataframe_to_dmatrix(tt,col)
model    = xgb.train(xgb_params,dtr,num_boost_round=opt_n_booster)
price_tr_hat  = np.exp(model.predict(dtr)) - 1
price_tt_hat  = np.exp(model.predict(dtt)) - 1
if np.abs(expt_results_extra.iloc[0].tr - np.sqrt(np.mean((np.log(tr.price_doc+1)-np.log(price_tr_hat+1))**2))) < 10**-7: print "OK"
else: print "Warning: metric mismatch"

tt['price_doc']  = price_tt_hat
submit(tt,"20170502_x_%s" % str(expt_results_extra.index[0]))
