from common import *

def get_basic_features(d):
    d['year']     = pd.to_numeric(d.timestamp.str[:4])
    d['ym']       = d.timestamp.str[:7]
    #
    d['life_ratio']    = np.where(d.full_sq>0,d.life_sq/d.full_sq,np.nan)
    d['floor_ratio']   = np.where(d.max_floor>0,d.floor/d.max_floor,np.nan)
    for x in range(1,7): d["material=%d" % x] = d.material==x
    d['build_age']     = np.where((d.build_year>1000)&(d.build_year<3000),pd.to_numeric(d.timestamp.str[:4])-d.build_year,np.nan)
    d['room_size']     = np.where(d.num_room>0,d.full_sq/d.num_room,np.nan)
    d['kitch_ratio']   = np.where(d.full_sq>0,d.kitch_sq/d.full_sq,np.nan)
    for x in range(1,5): d["state=%d" % x] = d.state==x
    d['is_investment']    = d.product_type=='Investment'
    d['is_owneroccupier'] = d.product_type=='OwnerOccupier'
    #
    d['monthly_volume']  = d.groupby('ym').id.count()[d.ym].values
    d['monthly_volume_invest']  = d[d.is_investment].groupby('ym').id.count()[d.ym].fillna(0).values
    d['monthly_volume_occupy']  = d[d.is_owneroccupier].groupby('ym').id.count()[d.ym].fillna(0).values
    #
    col_sets = [
        ['full_sq','life_sq','life_ratio','floor','max_floor','floor_ratio'],
        ['has_additional_info'],
        ["material=%d" % x for x in range(1,7)],
        ['build_year','build_age','num_room','room_size','kitch_sq','kitch_ratio','state'],
        ["state=%d" % x for x in range(1,5)],
        ['is_investment','is_owneroccupier'],
        ['monthly_volume','monthly_volume_invest','monthly_volume_occupy']]
    col  = [x for y in col_sets for x in y]
    return d,col

def get_raion_features(d):
    d['raion_pop_density'] = d.raion_popul / d.area_m
    #
    d['raion_volume']  = d.groupby('sub_area').id.count()[d.sub_area].values
    d['raion_volume_invest']  = d[d.product_type=='Investment'].groupby('sub_area').id.count()[d.sub_area].fillna(0).values
    d['raion_volume_occupy']  = d[d.product_type=='OwnerOccupier'].groupby('sub_area').id.count()[d.sub_area].fillna(0).values
    #
    d['raion_build_count_with_material_info_pc'] = d.raion_build_count_with_material_info / d.raion_popul
    d['raion_build_count_with_builddate_info_pc'] = d.raion_build_count_with_builddate_info / d.raion_popul
    #
    col_yesno  = ['culture_objects_top_25','thermal_power_plant_raion','incineration_raion','oil_chemistry_raion','radiation_raion','railroad_terminal_raion','big_market_raion','nuclear_reactor_raion','detention_facility_raion']
    for col in col_yesno:  d[col+'_yes']  = (d[col]=='yes')
    #
    d['female_ratio']  = d.female_f / d.full_all
    for name in ['young','work','ekder','0_6','7_14','0_17','16_29','0_13']:
        d[name+'_female_ratio']  = d[name+'_female'] / d[name+'_all']
    #
    for name in ['block','wood','frame','brick','monolith','panel','foam','slag','mix']:
        d['build_ratio_'+name]  = d['build_count_'+name] / d['raion_build_count_with_material_info']
    for name in ['before_1920','1921-1945','1946-1970','1971-1995','after_1995']:
        d['build_ratio_'+name]  = d['build_count_'+name] / d['raion_build_count_with_builddate_info']
    #
    col  = ['area_m','raion_popul','raion_pop_density',
        'raion_volume','raion_volume_invest','raion_volume_occupy',
        'green_zone_part','indust_part',
        'children_preschool','preschool_quota','preschool_education_centers_raion',
        'children_school','school_quota','school_education_centers_raion',
        'school_education_centers_top_20_raion',
        'hospital_beds_raion','healthcare_centers_raion',
        'university_top_20_raion','sport_objects_raion','additional_education_raion',
        'culture_objects_top_25_yes','culture_objects_top_25_raion',
        'shopping_centers_raion','office_raion',
        'thermal_power_plant_raion_yes','incineration_raion_yes','oil_chemistry_raion_yes','radiation_raion_yes',
        'railroad_terminal_raion_yes','big_market_raion_yes',
        'nuclear_reactor_raion_yes','detention_facility_raion_yes',
        'full_all','female_ratio',
        'young_all','young_female_ratio',
        'work_all','work_female_ratio',
        'ekder_all','ekder_female_ratio',
        '0_6_all','0_6_female_ratio',
        '7_14_all','7_14_female_ratio',
        '0_17_all','0_17_female_ratio',
        '16_29_all','16_29_female_ratio',
        '0_13_all','0_13_female_ratio',
        'raion_build_count_with_material_info',
        'raion_build_count_with_material_info_pc',
        # 'build_count_block','build_count_wood','build_count_frame','build_count_brick','build_count_monolith','build_count_panel','build_count_foam','build_count_slag','build_count_mix',
        'build_ratio_block','build_ratio_wood','build_ratio_frame','build_ratio_brick','build_ratio_monolith','build_ratio_panel','build_ratio_foam','build_ratio_slag','build_ratio_mix',
        'raion_build_count_with_builddate_info',
        'raion_build_count_with_builddate_info_pc',
        # 'build_count_before_1920','build_count_1921-1945','build_count_1946-1970','build_count_1971-1995','build_count_after_1995',
        'build_ratio_before_1920','build_ratio_1921-1945','build_ratio_1946-1970','build_ratio_1971-1995','build_ratio_after_1995']
    return d,col

def get_ID_median_prices(tr,target='price_doc'):
    ID_col = ['ID_metro','ID_railroad_station_walk','ID_railroad_station_avto','ID_big_road1','ID_big_road2','ID_railroad_terminal','ID_bus_terminal']
    tr['price_per_sq']  = tr[target]/ tr.full_sq
    ID_prices = {}
    for col in ID_col:
        ID_prices[col]  = tr.groupby(col)[[target,'price_per_sq']].median().rename(columns={target:'price'}).sort_values('price',ascending=False)
    return ID_prices

def get_neighborhood_features(d,ID_prices):
    col_yesno  = ['water_1line','big_road1_1line','railroad_1line']
    for col in col_yesno:  d[col+'_yes']  = (d[col]=='yes')
    #
    # col_cat    = ['ID_metro_thd','ID_railroad_station_walk_thd','ID_railroad_station_avto_thd','ID_big_road1_thd','ID_big_road2_thd','ID_railroad_terminal_thd','ID_bus_terminal_thd','ecology']
    # d  = pd.concat([d,pd.get_dummies(d[col_cat],columns=col_cat,prefix=col_cat,prefix_sep='=')],1)
    d['ecology=no data']  = d.ecology=='no data'
    #
    col_ID  = ['ID_metro','ID_railroad_station_walk','ID_railroad_station_avto','ID_big_road1','ID_big_road2','ID_railroad_terminal','ID_bus_terminal']
    for col in col_ID:
        d[col+'_volume']  = d.groupby(col).id.count()[d[col]].values
        d[col+'_volume_invest']  = d[d.is_investment].groupby(col).id.count()[d[col]].fillna(0).values
        d[col+'_volume_occupy']  = d[d.is_owneroccupier].groupby(col).id.count()[d[col]].fillna(0).values
        d[col+'_price']          = ID_prices[col].price[d[col]].values
        d[col+'_price_per_sq']   = ID_prices[col].price_per_sq[d[col]].values
    #
    col_sets  = [
        [x+'_'+name for x in ['ID_metro','ID_railroad_station_walk','ID_railroad_station_avto','ID_big_road1','ID_big_road2','ID_railroad_terminal','ID_bus_terminal'] for name in ['volume','volume_invest','volume_occupy','price','price_per_sq']],
        # ["ID_metro_thd=%.1f" % name for name in get_dummy_values(tt,'ID_metro',99)],
        # ["ID_railroad_station_walk_thd=%.1f" % name for name in get_dummy_values(tt,'ID_railroad_station_walk',200)],
        # ["ID_railroad_station_avto_thd=%.1f" % name for name in get_dummy_values(tt,'ID_railroad_station_avto',200)],
        ['metro_min_avto','metro_km_avto','metro_min_walk','metro_km_walk','railroad_station_walk_km','railroad_station_walk_min','railroad_station_avto_km','railroad_station_avto_min','public_transport_station_km','public_transport_station_min_walk'],
        ['kindergarten_km','school_km','park_km','green_zone_km','industrial_km','water_treatment_km','cemetery_km','incineration_km'],
        ['water_km','water_1line_yes'],
        ['mkad_km','ttk_km','sadovoe_km','bulvar_ring_km','kremlin_km'],
        # ["ID_big_road1_thd=%.1f" % name for name in get_dummy_values(tt,'ID_big_road1',300)],
        # ["ID_big_road2_thd=%.1f" % name for name in get_dummy_values(tt,'ID_big_road2',300)],
        ['big_road1_km','big_road1_1line_yes','big_road2_km','railroad_km','railroad_1line_yes','zd_vokzaly_avto_km','bus_terminal_avto_km'],
        # ["ID_railroad_terminal_thd=%.1f" % name for name in get_dummy_values(tt,'ID_railroad_terminal',300)],
        # ["ID_bus_terminal_thd=%.1f" % name for name in get_dummy_values(tt,'ID_bus_terminal',500)],
        ['oil_chemistry_km','nuclear_reactor_km','radiation_km','power_transmission_line_km','thermal_power_plant_km','ts_km'],
        ['big_market_km','market_shop_km','fitness_km','swim_pool_km','ice_rink_km','stadium_km','basketball_km'],
        ['hospice_morgue_km','detention_facility_km'],
        ['public_healthcare_km','university_km','workplaces_km','shopping_centers_km','office_km','additional_education_km','preschool_km','big_church_km','church_synagogue_km','mosque_km','theater_km','museum_km','exhibition_km','catering_km'],
        ['ecology=no data'], #['ecology='+name for name in tt.ecology.unique()],
        [x%y for y in [500,1000,1500,2000,3000,5000] for x in ['green_part_%d','prom_part_%d','office_count_%d','office_sqm_%d','trc_count_%d','trc_sqm_%d','cafe_count_%d','cafe_sum_%d_min_price_avg','cafe_sum_%d_max_price_avg','cafe_avg_price_%d','cafe_count_%d_na_price','cafe_count_%d_price_500','cafe_count_%d_price_1000','cafe_count_%d_price_1500','cafe_count_%d_price_2500','cafe_count_%d_price_4000','cafe_count_%d_price_high','big_church_count_%d','church_count_%d','mosque_count_%d','leisure_count_%d','sport_count_%d','market_count_%d']]]
    col  = [x for y in col_sets for x in y]
    return d,col

def get_features(d,ID_prices):
    d,feat_basic  = get_basic_features(d)
    d,feat_raion  = get_raion_features(d)
    d,feat_neighborhood = get_neighborhood_features(d,ID_prices)
    feat_macro    = ['cpi','ppi','usdrub','eurrub','fixed_basket','rent_price_4+room_bus','rent_price_3room_bus','rent_price_2room_bus','rent_price_1room_bus','rent_price_3room_eco','rent_price_2room_eco','rent_price_1room_eco']
    #
    feat  = feat_basic + feat_raion + feat_neighborhood + feat_macro
    #
    return d,feat

tr  = pd.read_csv('train.csv')
tt  = pd.read_csv('test.csv')
tt['material']  = tt.material.astype(float)

macro = pd.read_csv('macro.csv')
macro_col  = ['cpi','ppi','usdrub','eurrub','fixed_basket','rent_price_4+room_bus','rent_price_3room_bus','rent_price_2room_bus','rent_price_1room_bus','rent_price_3room_eco','rent_price_2room_eco','rent_price_1room_eco']
tr  = tr.merge(macro[['timestamp']+macro_col],how='left',on='timestamp',copy=False)
tt  = tt.merge(macro[['timestamp']+macro_col],how='left',on='timestamp',copy=False)

clean_data(tr)
clean_data(tt)

validation_partition_date = '2014-07-01'
ID_prices_tr1  = get_ID_median_prices(tr[tr.timestamp<validation_partition_date])
ID_prices_tr   = get_ID_median_prices(tr)

tt,col  = get_features(tt,ID_prices_tr)
dtt     = dataframe_to_dmatrix(tt,col)

exit(0)

#-- 1st Round optimize 'max_depth','min_child_weight','gamma' --#
expt  = pd.DataFrame(
    columns=('tr1','vn','vnopt','optn','tr','prjtt'),
    index=pd.MultiIndex.from_product(
        [[validation_partition_date],range(2,8),range(1,8,2),[0,0.1,0.2,0.4],[1],[0.5],[0.25],[8],[0.2],[30]],
        names=('vn_part_date','max_depth','min_child_weight','gamma','subsample','colsample_bytree','lambda','alpha','eta','early_stop')))
best_vn       = np.inf
for idx in expt.index:
    t0  = time.clock()
    xgb_params  = idx_to_xgb_params(idx,expt.index.names)
    #
    tr,col  = get_features(tr,ID_prices_tr1)
    opt_nboost,tv_evals = xgb_temporal_validation(xgb_params,tr,col)
    expt.loc[idx,['tr1','vn','optn']]  = [tv_evals['tr1']['rmse'][opt_nboost-1],tv_evals['vn']['rmse'][opt_nboost-1],opt_nboost]
    #
    if best_vn > expt.loc[idx,'vn']:
        best_vn    = expt.loc[idx,'vn']
        model      = xgb.train(xgb_params,dataframe_to_dmatrix(tr[tr.timestamp<validation_partition_date],col),num_boost_round=opt_nboost)
        feat_rank  = xgb_feat_rank(model,col)
    #
    tr,col = get_features(tr,ID_prices_tr)
    dtr    = dataframe_to_dmatrix(tr,col)
    model  = xgb.train(xgb_params,dtr,num_boost_round=opt_nboost)
    price_tr_hat  = np.exp(model.predict(dtr)) - 1
    expt.loc[idx,'tr']  = np.sqrt(np.mean((np.log(tr.price_doc+1)-np.log(price_tr_hat+1))**2))
    #
    print idx
    print expt.loc[[idx]].reset_index(drop=True)
    print (time.clock()-t0)

expt['vnopt']  = expt.vn - expt.tr1
expt['prjtt']  = expt.tr + expt.vnopt
expt.to_pickle('xgb_optparam_20170503_1st.pkl')

print expt.sort_values('vn')
expt.to_clipboard('\t')
feat_rank.to_clipboard('\t')


expt = pd.read_pickle('xgb_optparam_20170503_1st.pkl').sort_values('vn').iloc[:3]
ID_prices_tr   = get_ID_median_prices(tr)
for i in range(3):
    tr,col = get_features(tr,ID_prices_tr)
    dtr    = dataframe_to_dmatrix(tr,col)
    xgb_params  = idx_to_xgb_params(expt.index[i],expt.index.names)
    model       = xgb.train(xgb_params,dtr,num_boost_round=expt.iloc[i].optn)
    tr_price_hat  = np.exp(model.predict(dtr)) - 1
    if np.abs(np.sqrt(np.mean((np.log(tr_price_hat+1)-np.log(tr.price_doc+1))**2))-expt.iloc[i].tr) < 10**-7: print "OK"
    else: print "metric mismatch!!!"
    tt,col  = get_features(tt,ID_prices_tr)
    dtt     = dataframe_to_dmatrix(tt,col)
    tt_price_hat  = np.exp(model.predict(dtt)) - 1
    tt['price_doc']  = tt_price_hat
    submit(tt,"20170503_%d_%s" % (i+1,str(expt.index[i])))
#
expt_adj  = {}
expt_adj['none']  = expt
for adj in ['cpi','ppi','fixed_basket']:
    expt_adj[adj]  = expt.copy()
    #
    adj_mean_tr  = tr[adj].mean()
    adj_mean_all = pd.concat([tr[adj],tt[adj]]).mean()
    #
    for i,idx in enumerate(expt_adj[adj].index):
        xgb_params  = idx_to_xgb_params(idx,expt_adj[adj].index.names)
        #
        #-- Validation to find OptNBoost
        tr['price_'+adj]  = tr.price_doc * adj_mean_tr / tr[adj]
        ID_prices_adj_tr1 = get_ID_median_prices(tr[tr.timestamp<validation_partition_date],target='price_'+adj)
        tr,col  = get_features(tr,ID_prices_adj_tr1)
        opt_nboost,_ = xgb_temporal_validation(xgb_params,tr,col,target='price_'+adj)
        #
        #-- Calculate validation scores
        tr1   = tr[tr.timestamp<validation_partition_date]
        vn    = tr[tr.timestamp>=validation_partition_date]
        dtr1  = dataframe_to_dmatrix(tr1,col,target='price_'+adj)
        dvn   = dataframe_to_dmatrix(vn,col,target='price_'+adj)
        model = xgb.train(xgb_params,dtr1,num_boost_round=opt_nboost)
        tr1_price_hat = (np.exp(model.predict(dtr1)) - 1) * tr1[adj] / adj_mean_tr
        vn_price_hat  = (np.exp(model.predict(dvn)) - 1) * vn[adj] / adj_mean_tr
        expt_adj[adj].loc[idx,['tr1','vn','optn']] = [
            np.sqrt(np.mean((np.log(tr1_price_hat+1)-np.log(tr1.price_doc+1))**2)),
            np.sqrt(np.mean((np.log(vn_price_hat+1)-np.log(vn.price_doc+1))**2)),
            opt_nboost]
        #
        #-- Train model on all training data and predict tt
        tr['price_'+adj]  = tr.price_doc * adj_mean_all / tr[adj]
        ID_prices_adj_tr  = get_ID_median_prices(tr,target='price_'+adj)
        tr,col  = get_features(tr,ID_prices_adj_tr)
        dtr     = dataframe_to_dmatrix(tr,col,target='price_'+adj)
        model   = xgb.train(xgb_params,dtr,num_boost_round=opt_nboost)
        tr_price_hat  = (np.exp(model.predict(dtr)) - 1) * tr[adj] / adj_mean_all
        expt_adj[adj].loc[idx,'tr'] = np.sqrt(np.mean((np.log(tr_price_hat+1)-np.log(tr.price_doc+1))**2))
        #
        tt,col  = get_features(tt,ID_prices_adj_tr)
        dtt     = dataframe_to_dmatrix(tt,col,target='price_'+adj)
        tt_price_hat  = (np.exp(model.predict(dtt)) - 1) * tt[adj] / adj_mean_all
        tt['price_doc']  = tt_price_hat
        submit(tt,"20170503_%d_%s_%s" % (i+1,adj,str(idx)))

pd.concat(expt_adj.values(),keys=expt_adj.keys()).to_clipboard('\t')

