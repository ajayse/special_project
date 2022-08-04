# -*- coding: utf-8 -*-
"""price_comparison.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1RR_vJlSeUtt7OpR1kDu9fBS99lAfC0-D
"""

import pandas as pd
import streamlit as st
from st_aggrid import AgGrid
from st_aggrid.grid_options_builder import GridOptionsBuilder
from st_aggrid import GridUpdateMode, DataReturnMode, JsCode
import math
import numpy as np

st.set_page_config(page_icon=":chart_with_upwards_trend:", 
                   page_title="Gulong Price Comparison",
                   layout="wide",
                   menu_items = {
                       'About':"# Thanks for using the app!\n For inquiries, send an email to Arvin Escolano at arvinjayescolano.licagroup@gmail.com."
                       })
@st.experimental_memo
def acquire_data():
    url1, url2 =  "http://app.redash.licagroup.ph/api/queries/82/results.csv?api_key=fEm6wVpPGIH8R5Pks7I62ULYuVEOSZBBHHJgVwX4", "http://app.redash.licagroup.ph/api/queries/72/results.csv?api_key=aUZHO0183ucPVY4txOPrWiiiQPcxmjtiFFpwZ3ct"
    df_data = pd.read_csv(url1, parse_dates = ['supplier_price_date_updated','product_price_date_updated'])
    df_data = df_data[['make','model','slug', 'name','cost','srp', 'promo', 'supplier_price_date_updated','product_price_date_updated']]#'product_price_date_updated'
    df_data.columns = ['make','model', 'slug','supplier','supplier_price','gulong_price','promo_price','supplier_updated','gulong_updated']
    df_supplier = df_data[['make','model','supplier','supplier_price','supplier_updated']].copy().sort_values(by='supplier_updated',ascending=False)
    df_supplier = df_supplier.dropna()
    df_supplier = df_supplier.loc[df_supplier['supplier_price']>0]
    df_supplier = df_supplier.drop_duplicates(subset=['model','supplier'],keep='first')
    df_supplier = df_supplier.groupby(['make','model','supplier'], group_keys=False).agg(price = ('supplier_price',lambda x: x))
    df_supplier = df_supplier.unstack('supplier').reset_index().set_index(['make','model'])
    df_supplier.columns = [i[1] for i in df_supplier.columns]
    df_gulong = df_data[['make','model', 'gulong_price','promo_price','gulong_updated']].copy().sort_values(by='gulong_updated',ascending=False)
    df_gulong = df_gulong.drop_duplicates(subset='model',keep='first')
    df_gulong = df_gulong.loc[df_gulong['gulong_price']>1]
    df_gulong = df_gulong.drop('gulong_updated',axis = 1)
    df_gulong = df_gulong.set_index(['make','model'])
    df_final = df_gulong.join(df_supplier,how='outer').sort_values(by=['make','model']).reset_index()
    return df_final, df_supplier.columns


def convert_df(df):
     # IMPORTANT: Cache the conversion to prevent computation on every rerun
     return df.to_csv().encode('utf-8')

def consider_GP(data,GP):
    return ceil_5(data/(1-GP/100))

def ceil_5(n):
    return math.ceil(n/5)*5

def radio_GP(price, tier):
    if tier =="1":
        return ceil_5(price/(1-0.30))
    if tier =="2":
        return ceil_5(price/(1-0.27))
    if tier =="3":
        return round(price/(1-0.25),2)
    if tier =="4":
        return ceil_5(price/(1-0.28))
    if tier =="5":
        return round(price/(1-0.27),2)

def apply_tier(df):
    tiers = []
    if tier1:
        df['Tier 1'] = df['max_price'].apply(lambda x: radio_GP(x,"1"))
        tiers.append('Tier 1')
    if tier2:
        df['Tier 2'] = df['max_price'].apply(lambda x: radio_GP(x,"2"))
        tiers.append('Tier 2')
    if tier3:
        df['Tier 3'] = df['max_price'].apply(lambda x: radio_GP(x,"3"))
        tiers.append('Tier 3')
    if tier4:
        df['Tier 4'] = df['max_price'].apply(lambda x: radio_GP(x,"4"))
        tiers.append('Tier 4')
    if tier5:
        df['Tier 5'] = df['max_price'].apply(lambda x: radio_GP(x,"5"))
        tiers.append('Tier 5')
    return df,tiers

def highlight_gulong(x):
    c1 = (x['gulong_price'] != x.max(axis=1))
    df1 = pd.DataFrame('background-color: ', index=x.index, columns=x.columns)
    df1['gulong_price']= np.where(c1, 'background-color: {}'.format('pink'), df1['gulong_price'])
    return df1

def highlight_others(x):#cols = ['GP','Tier 1','Tier 3', etc]
    cols = ['GP']
    cols.extend(tiers)
    df1 = pd.DataFrame('background-color: ', index=x.index, columns=x.columns)
    for column in cols: 
        c = (x[column] > x['gulong_price'])
        df1[column]= np.where(c, 'color:{};font-weight:{}'.format('red','bold'), df1[column])#.set_index(['make','model'])
    return df1

def return_suppliers():
    selected_supplier_ = list(supplier_cols)

df_final, supplier_cols = acquire_data()



st.header("All Data")
cols = ['make','model','gulong_price','promo_price']

with st.expander('Include/remove suppliers in list:'):
    beta_multiselect = st.container()
    check_all = st.checkbox('Select all', value=True)
    if check_all:
        selected_supplier_ = beta_multiselect.multiselect('Included suppliers in table:',
                                       options = supplier_cols,
                                       default = list(supplier_cols))
    else:
        selected_supplier_ = beta_multiselect.multiselect('Included suppliers in table:',
                                       options = supplier_cols)

cols.extend(selected_supplier_)
st.write("Select the SKUs that would be considered for the computations.",
         " Feel free to filter the _make_ and _model_ that would be shown. You may also select/deselect columns.")

df_show =df_final[cols].dropna(how = 'all',subset = selected_supplier_,axis=0).fillna(0)
df_show = df_show.replace(0,'')

gb = GridOptionsBuilder.from_dataframe(df_show)
gb.configure_default_column(enablePivot=False, enableValue=False, enableRowGroup=False, editable = True)
gb.configure_column('make', headerCheckboxSelection = True)
gb.configure_selection(selection_mode="multiple", use_checkbox=True)
gb.configure_side_bar()  
gridOptions = gb.build()

response = AgGrid(df_show,
    theme = 'light',
    gridOptions=gridOptions,
    height = 300,
    #width = '100%',
    editable=True,
    allow_unsafe_jscode=True,
    reload_data=False,
    enable_enterprise_modules=True,
    update_mode=GridUpdateMode.MODEL_CHANGED,
    data_return_mode=DataReturnMode.FILTERED_AND_SORTED,
    fit_columns_on_grid_load=False
)

st.write("Results: "+str(len(df_show))+" entries")

st.markdown("""
            ---
            """)
st.header("Price Comparison")
st.write("You may set the GP and the price comparison between models would be shown in a table.")
c1, cs1,cS, ct1, ct2,ct3,ct4,ct5,cs2 = st.columns([2,2,1,1,1,1,1,1,3])
with c1:
    GP = st.number_input("GP (%):",min_value=0.00,max_value = 100.00,
                              value=25.00, step = 0.01)
with cS:
    st.write('Include tier:')
with ct1:
    tier1 = st.checkbox('Tier 1')
with ct2:
    tier2 = st.checkbox('Tier 2')
with ct3:
    tier3 = st.checkbox('Tier 3')
with ct4:
    tier4 = st.checkbox('Tier 4')
with ct5:
    tier5 = st.checkbox('Tier 5')

df = pd.DataFrame.from_dict(response['selected_rows'])

if len(df)>0:
    df = df.drop('make', axis =1)
    df = df.set_index('model')
    df = df.replace('',np.nan).dropna(axis=1, how='all')
    if 'rowIndex' in df.columns.to_list():
        df = df.drop('rowIndex', axis =1)
    
    df['max_price'] = df[df.columns.to_list()[2:]].fillna(0).apply(lambda x: round(x.max(),2),axis=1)
    df['GP'] = df[df.columns.to_list()[-1]].apply(lambda x: consider_GP(x,GP))
    df,tiers = apply_tier(df)
    st.dataframe(df.style.apply(highlight_gulong, axis=None)\
             .apply(highlight_others,axis=None)\
             .format(precision = 2))
             #.format(formatter={"max_price": "{:.2f}", "Tier 3": "{:.2f}","Tier 5": "{:.2f}"}))

else:
    st.info("Kindly check/select at lease one row above.")

st.markdown("""
            ---
            """)
            
st.header('Download tables:')
st.write("**All data:**")
if df_final is not None:
    csv = convert_df(df_show)
    st.download_button(
                        label="gulong_pricing.csv",
                        data=csv,
                        file_name='gulong_pricing.csv',
                        mime='text/csv'
                        )
if df is not None:
    csv = convert_df(df)
    st.write("**Price comparison table:**")
    st.download_button(
                        label="price_comparison.csv",
                        data=csv,
                        file_name='price_comparison.csv',
                        mime='text/csv'
                        )
    
st.markdown("""
            ---
            """)

cA, cB =st.columns([2,2])
with cA:
    st.warning('Update data only if neccessary')
with cB:
    if st.button('Update Data'):
        st.experimental_memo.clear()
        df_final = pd.DataFrame()
        supplier_cols = []
        



#elif view =='Select by Supplier':
    

# GP = st.sidebar.number_input("GP (%):",min_value=0.00,max_value = 100.00,
#                              value=25.00, step = 0.01)

# supplier_list = st.sidebar.multiselect("Select supplier/s", supplier_cols)
# st.sidebar.write("Include tier:")
# t1 = st.sidebar.checkbox("Tier 1")
# t2 = st.sidebar.checkbox("Tier 2")
# t3 = st.sidebar.checkbox("Tier 3")
# t4 = st.sidebar.checkbox("Tier 4")
# t5 = st.sidebar.checkbox("Tier 5")

# st.write()







# cC, cD, cE = st.columns([1, 8, 1])
# with cD:
#     if df_final is not None:
#         raw_container = st.expander('Show uploaded data:')
#         raw_container.write(df_final)
