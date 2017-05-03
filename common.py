import time
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import xgboost as xgb
import scipy.stats as stats

quantiles = [0,0.01,0.05,0.25,0.5,0.75,0.95,0.99,1]

def clean_data(d):
    d['full_sq']  = np.where(d.full_sq>=10,d.full_sq,np.where(d.life_sq>=10,d.life_sq,np.nan)) # Cleaned version show slightly larger correlation with price_doc
    d['life_sq']  = np.where(d.life_sq>1,d.life_sq,np.nan) # Cleaned version show slightly larger correlation with price_doc
    d.loc[(d.floor==0) | (d.floor==77),'floor']  = np.nan # Training contains entry with floor=77 and max floor=22 #
    d.loc[d.max_floor==0,'max_floor'] = np.nan
    d.loc[(d.floor>d.max_floor)&(d.max_floor<=1),'max_floor'] = np.nan
    # max_floor correlation with price increased after cleaning
    # Fields 'max_floor', 'material', 'build_year', 'num_room', 'kitch_sq', 'state' is mostly missing until 2013.7, and mostly filled after 2013.9
    # d.loc[d.id==13120,'build_year']  = d.loc[d.id==13120,'kitch_sq']
    d.loc[(d.build_year<1000) | (d.build_year>3000),'build_year'] = np.nan
    d['has_additional_info']  = d.num_room.notnull()
    d.loc[d.num_room==0,'num_room']  = np.nan
    d.loc[(d.kitch_sq>500) | (d.kitch_sq<=1),'kitch_sq']  = np.nan
    d.loc[d.state>4,'state'] = np.nan

def xgb_temporal_validation(xgb_params,tr,col,target='price_doc'):
    baseline_rmsle = np.sqrt(np.mean((np.log(tr.loc[tr.timestamp>=xgb_params['vn_part_date'],target]+1) - np.log(tr.loc[tr.timestamp<xgb_params['vn_part_date'],target]+1).mean())**2))
    print "Baseline RMSLE = %.5f" % baseline_rmsle
    dtr  = dataframe_to_dmatrix(tr[tr.timestamp<xgb_params['vn_part_date']],col,target=target)
    dvn  = dataframe_to_dmatrix(tr[tr.timestamp>=xgb_params['vn_part_date']],col,target=target)
    tv_evals  = {}
    tv_model  = xgb.train(xgb_params,dtr,num_boost_round=1000,evals=[(dtr,'tr1'),(dvn,'vn')],early_stopping_rounds=xgb_params['early_stop'],evals_result=tv_evals,verbose_eval=False)
    opt_nboost = tv_model.best_ntree_limit
    print "Training RMSLE = %.5f" % tv_evals['tr1']['rmse'][opt_nboost-1]
    print "Validation RMSLE = %.5F" % tv_evals['vn']['rmse'][opt_nboost-1]
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

def pearsonr_w_price(d,col,col_price='price_doc'):
    mask  = d[col].notnull() & d[col_price].notnull()
    x     = d.loc[mask,col]
    y     = d.loc[mask,col_price]
    logy  = np.log(y)
    return stats.pearsonr(x,y),stats.pearsonr(x,logy)

def dataframe_to_dmatrix(d,col,target='price_doc'):
    return xgb.DMatrix(d[col].fillna(-1),label=np.log(d[target] + 1) if target in d else None)

def submit(tt,name):
    tt[['id','price_doc']].to_csv("submission_%s.csv" % name,index=False)

def get_dummy_values(d,col,thd):
    c  = d[col].value_counts()
    return c[c>thd].index.tolist()

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

def get_ID_median_prices(tr):
    ID_col = ['ID_metro','ID_railroad_station_walk','ID_railroad_station_avto','ID_big_road1','ID_big_road2','ID_railroad_terminal','ID_bus_terminal']
    tr['price_per_sq']  = tr.price_doc / tr.full_sq
    ID_prices = {}
    for col in ID_col:
        ID_prices[col]  = tr.groupby(col)[['price_doc','price_per_sq']].median().rename(columns={'price_doc':'price'}).sort_values('price',ascending=False)
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
