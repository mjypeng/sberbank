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

def submit(tt,name):
    tt[['id','price_doc']].to_csv("submission_%s.csv" % name,index=False)

tr  = pd.read_csv('train.csv')
tt  = pd.read_csv('test.csv')
tt['material']  = tt.material.astype(float)

# col_rgnsta  = [] # Columns stationary and fully dependent on sub_area (raion?)
# for col in tr:
#     print col
#     if np.all(pd.concat([tr,tt]).fillna(-9999).groupby('sub_area')[col].nunique()==1):
#         col_rgnsta.append(col)

# pd.concat([tr[col_rgnsta].drop_duplicates().set_index('sub_area'),tr.groupby('sub_area')

# pd.concat([tr,tt])[col_rgnsta].drop_duplicates().set_index('sub_area').to_csv('raion_info.csv')

raion = pd.read_csv('raion_info.csv')

macro = pd.read_csv('macro.csv')
tr  = tr.merge(macro[['timestamp','cpi','ppi']],how='left',on='timestamp',copy=False)
tt  = tt.merge(macro[['timestamp','cpi','ppi']],how='left',on='timestamp',copy=False)
tr['adj_rub_cpi']  = tr.cpi.mean() / tr.cpi
tr['adj_rub_ppi']  = tr.ppi.mean() / tr.ppi
tt['adj_rub_cpi']  = tt.cpi.mean() / tt.cpi
tt['adj_rub_ppi']  = tt.ppi.mean() / tt.ppi

col_cat_thd  = {'sub_area':80, 'ID_metro':99, 'ID_railroad_station_walk':200, 'ID_railroad_station_avto':200, 'ID_big_road1':300, 'ID_big_road2':300, 'ID_railroad_terminal':300, 'ID_bus_terminal':500}
for col in col_cat_thd:
    tr[col+'_thd'] = np.where(tr[col].isin(get_dummy_values(tt,col,col_cat_thd[col])),tr[col],np.nan)
    tt[col+'_thd'] = np.where(tt[col].isin(get_dummy_values(tt,col,col_cat_thd[col])),tt[col],np.nan)

tr  = preprocess(tr)
tt  = preprocess(tt)

tr  = tr[tr.sq_inferred.notnull()]
tr['price_per_sq']     = tr.price_doc/tr.sq_inferred
tr['adj_price_per_sq'] = tr.price_per_sq * (tr.adj_rub_cpi + tr.adj_rub_ppi)/2

temp = tr[['sub_area','price_per_sq','adj_price_per_sq']].groupby('sub_area').agg([len,np.median,lambda x:np.subtract(*x.quantile([0.75,0.25]))]).iloc[:,[0,1,2,4,5]]
temp.columns  = ['tr_vol','price_per_sq_median','price_per_sq_IQR','adj_price_per_sq_median','adj_price_per_sq_IQR']
temp2 = tt.groupby('sub_area')[['id']].count().rename(columns={'id':'tt_vol'})
raion = pd.concat([temp,temp2,raion.set_index('sub_area')],1)


# Rows with suspected invalid 'full_sq' and 'life_sq'
tr[tr.sq_inferred==10].to_clipboard('\t',index=False)
tt[((tt.sq_inferred<10)|tt.sq_inferred.isnull())].to_clipboard('\t',index=False)

tr[['sub_area','price_per_sq','cpi','ppi']].groupby('sub_area').median().to_clipboard('\t')

# tr  = tr[(tr.price_per_sq>0) & (tr.price_per_sq<np.inf)]

# tr.price_doc.quantile([0,0.01,0.05,0.25,0.5,0.75,0.95,0.99,1])
tr.price_per_sq.quantile([0,0.01,0.05,0.25,0.5,0.75,0.95,0.99,1])

# tr.full_sq.quantile([0,0.01,0.05,0.25,0.5,0.75,0.95,0.99,1])


