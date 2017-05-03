from common import *

def quantile(d,col):
    q  = d[d[col].notnull()].groupby(level=0)[col].quantile(quantiles)
    return pd.concat([q.loc['tr'],q.loc['tt']],1,keys=('tr','tt'))

def value_counts(d,col):
    c  = d[col].fillna('Missing').groupby(level=0).value_counts()
    return pd.concat([c.loc['tr'],c.loc['tt']],1,keys=('tr','tt')).fillna(0).astype(int)

tr  = pd.read_csv('train.csv')
tt  = pd.read_csv('test.csv')
tt['material']  = tt.material.astype(float)

macro = pd.read_csv('macro.csv')
macro_col  = ['cpi','ppi','usdrub','eurrub','fixed_basket','rent_price_4+room_bus','rent_price_3room_bus','rent_price_2room_bus','rent_price_1room_bus','rent_price_3room_eco','rent_price_2room_eco','rent_price_1room_eco']
tr  = tr.merge(macro[['timestamp']+macro_col],how='left',on='timestamp',copy=False)
tt  = tt.merge(macro[['timestamp']+macro_col],how='left',on='timestamp',copy=False)
d  = pd.concat([tr,tt],0,keys=('tr','tt'))
d['ym'] = d.timestamp.str[:7]

index_col  = ['id','timestamp']
basic_col  = ['full_sq','life_sq','floor','max_floor','material','build_year','num_room','kitch_sq','state','product_type']
raion_col  = ['sub_area','area_m','raion_popul',
    'green_zone_part','indust_part',
    'children_preschool','preschool_quota','preschool_education_centers_raion',
    'children_school','school_quota','school_education_centers_raion',
    'school_education_centers_top_20_raion',
    'hospital_beds_raion','healthcare_centers_raion',
    'university_top_20_raion','sport_objects_raion','additional_education_raion',
    'culture_objects_top_25','culture_objects_top_25_raion',
    'shopping_centers_raion','office_raion',
    'thermal_power_plant_raion','incineration_raion','oil_chemistry_raion','radiation_raion',
    'railroad_terminal_raion','big_market_raion',
    'nuclear_reactor_raion','detention_facility_raion',
    'full_all','male_f','female_f',
    'young_all','young_male','young_female',
    'work_all','work_male','work_female',
    'ekder_all','ekder_male','ekder_female',
    '0_6_all','0_6_male','0_6_female',
    '7_14_all','7_14_male','7_14_female',
    '0_17_all','0_17_male','0_17_female',
    '16_29_all','16_29_male','16_29_female',
    '0_13_all','0_13_male','0_13_female',
    'raion_build_count_with_material_info',
    'build_count_block','build_count_wood','build_count_frame','build_count_brick','build_count_monolith','build_count_panel','build_count_foam','build_count_slag','build_count_mix',
    'raion_build_count_with_builddate_info',
    'build_count_before_1920','build_count_1921-1945','build_count_1946-1970','build_count_1971-1995','build_count_after_1995']

exit(0)

corr  = pd.DataFrame(columns=('r','p','r_log','p_log'))#,index=basic_col + macro_col)
for col in basic_col + macro_col:
    try:
        ((corr.loc[col,'r'],corr.loc[col,'p']),(corr.loc[col,'r_log'],corr.loc[col,'p_log'])) = pearsonr_w_price(tr[tr.has_additional_info],col,'price_doc')
    except:
        pass



tr,col1  = get_basic_features(tr)
tr,col2  = get_raion_features(tr)
col      = col1 + col2 + macro_col # # #col2 #  + 
xgb_params = {
    'vn_part_date': '2014-07-01',
    'early_stop': 30,
    'objective':'reg:linear',
    'eval_metric':'rmse',
    'silent':1,
    'booster':'gbtree',
      'eta':0.2,
      'max_depth':4,
      'min_child_weight':1,
      'subsample':0.8,
      'colsample_bytree':0.8,
      'lambda':1,
      'gamma':0.2,
      'alpha':0}
nboost,evals = xgb_temporal_validation(xgb_params,tr,col)
print evals['tr1']['rmse'][nboost-1],evals['vn']['rmse'][nboost-1]
# Basic:
    # Baseline RMSLE = 0.60815
    # Training RMSLE = 0.49063
    # Validation RMSLE = 0.46725
# Raion:
    # Baseline RMSLE = 0.60815
    # Training RMSLE = 0.56115
    # Validation RMSLE = 0.55890
# Macro:
    # Baseline RMSLE = 0.60815
    # Training RMSLE = 0.59448
    # Validation RMSLE = 0.59425
# Basic + Raion:
    # Baseline RMSLE = 0.60815
    # Training RMSLE = 0.45743
    # Validation RMSLE = 0.43775
# Basic + Macro:
    # Baseline RMSLE = 0.60815
    # Training RMSLE = 0.49615
    # Validation RMSLE = 0.47186
# Raion + Macro:
    # Baseline RMSLE = 0.60815
    # Training RMSLE = 0.54446
    # Validation RMSLE = 0.54304
# Basic + Raion + Macro:
    # Baseline RMSLE = 0.60815
    # Training RMSLE = 0.45443
    # Validation RMSLE = 0.43867

# Without clean_data: 0.496558 0.468478
# Clean full_sq: 0.49788 0.467621
# + Clean life_sq: 0.49559 0.467948
# + Clean floor: 0.49559 0.467986
# + Clean max_floor: 0.49651 0.467989
# + Impute build_year: 0.496491 0.468051
# + Clean build_year: 0.496597 0.467624
# + Clean num_room: 0.496813 0.467721
# + Clean kitch_sq: 0.53072 0.523942
# + Clean state: 0.53058 0.524197

# With clean data: 0.496399 0.467671

tr1  = tr[tr.timestamp<xgb_params['vn_part_date']]
vn   = tr[tr.timestamp>=xgb_params['vn_part_date']]
model  = xgb.train(xgb_params,dataframe_to_dmatrix(tr1,col,target='price_cpi'),num_boost_round=nboost)
price_adj_vn_hat  = np.exp(model.predict(dataframe_to_dmatrix(vn,col))) - 1
print np.sqrt(np.mean((np.log(price_adj_vn_hat+1) - np.log(vn.price_cpi+1))**2))
price_vn_hat  = price_adj_vn_hat*vn.cpi/tr.cpi.mean()
print np.sqrt(np.mean((np.log(price_vn_hat+1) - np.log(vn.price_doc+1))**2))


feat_rank  = xgb_feat_rank(model,col)
print feat_rank.sort_values('gain',ascending=False)

# Use d.num_room.notnull() to filter training data without all property features

'id','timestamp'

clean_data(tr)
tr['year']          = pd.to_numeric(tr.timestamp.str[:4])
tr['build_age']     = tr.year - tr.build_year
tr['price_per_sq']  = tr.price_doc/tr.full_sq

d[d.num_room.notnull()].groupby('ym')[['id','max_floor','material','build_year','num_room','kitch_sq','state']].count()

np.round(tr.groupby(['ID_metro','ID_railroad_station_walk','ID_railroad_station_avto','ID_big_road1','ID_big_road2','ID_railroad_terminal','ID_bus_terminal'])[['price_doc','price_per_sq']].agg([np.median,lambda x:np.subtract(*x.quantile([0.75,0.25]))])).to_clipboard('\t')


def get_ID_prices(tr):
    ID_col = ['ID_metro','ID_railroad_station_walk','ID_railroad_station_avto','ID_big_road1','ID_big_road2','ID_railroad_terminal','ID_bus_terminal']
    tr['price_per_sq']  = tr.price_doc / tr.full_sq
    ID_prices = {}
    for col in ID_col:
        ID_prices[col]  = tr.groupby(col)[['price_doc','price_per_sq']].median().rename(columns={'price_doc':'price'}).sort_values('price',ascending=False)


tt[ID_col].isin(tr[ID_col])


loc_tr = tr[].drop_duplicates().set_index(['ID_metro','ID_railroad_station_walk','ID_railroad_station_avto','ID_big_road1','ID_big_road2','ID_railroad_terminal','ID_bus_terminal'])
loc_tt = tt[['ID_metro','ID_railroad_station_walk','ID_railroad_station_avto','ID_big_road1','ID_big_road2','ID_railroad_terminal','ID_bus_terminal']].drop_duplicates().set_index(['ID_metro','ID_railroad_station_walk','ID_railroad_station_avto','ID_big_road1','ID_big_road2','ID_railroad_terminal','ID_bus_terminal'])
