import streamlit as st
import pandas as pd
import math
import json as simplejson
from st_aggrid import GridOptionsBuilder, AgGrid, GridUpdateMode, DataReturnMode


            
pd.options.display.max_rows = 99999
pd.set_option('max_colwidth', 400)
pd.options.mode.chained_assignment = None  # default='warn'

def fix_name(name):
  '''
  Fix names which are duplicated.
  Ex. "John Smith John Smith"
  Arguments:
    - name: str
  Returns:
    - fixed name; str
  '''
  name_list = list()
  for n in str.lower(name).split(' '):
    if n not in name_list:
      name_list.append(n)
  return ' '.join(name_list).strip()

def search_for_name(name):
  df_data.full_name = df_data.full_name.apply(lambda x: x.lower())
  names = df_data.loc[df_data.apply(lambda x: name.lower() in x.full_name, axis=1)]
  df_temp = names[['customer_id','full_name', 'brand', 'model/year','fuel_type','transmission','plate_number','phone','address','mileage','appointment_date','id','service_name']]
  df_temp['full_name'] = df_temp['full_name'].str.title()
  
  return df_temp.set_index('full_name')
  

all_data = pd.read_csv("http://app.redash.licagroup.ph/api/queries/128/results.csv?api_key=KisyFBTEg3GfiTZbrly189LJAHjwAzFIW7l9UElB", parse_dates = ['date','appointment_date','date_confirmed','date_cancelled'])
all_data = all_data.rename(columns={'year': 'model_year', 'name':'status'})
all_data = all_data[all_data['status']!='Cancelled']

all_data.loc[:,'full_name'] = all_data.apply(lambda x: fix_name(x['full_name']), axis=1)
all_data.loc[:,'brand'] = all_data.apply(lambda x: '' if x.empty else fix_name(x['brand']).upper(), axis=1)
all_data.loc[:,'model'] = all_data.apply(lambda x: '' if x.empty else  fix_name(x['model']).upper(), axis=1)
all_data['model_year'] = all_data['model_year'].apply(lambda x:  'XX' if math.isnan(x) else str(int(x)) )

all_data['model/year'] =all_data['model_year']+'/' + all_data['model'].str.upper()

cols = ['id', 'date', 'email','full_name','brand', 'model', 'model_year', 
        'appointment_date', 'mechanic_name', 'sub_total', 'service_fee', 'total_cost', 
        'date_confirmed', 'status', 'status_of_payment','customer_id','fuel_type','transmission','plate_number', 'phone','address','mileage','model/year']
drop_subset = ['full_name', 'brand', 'model', 'appointment_date','customer_id']
all_data_ = all_data[cols].drop_duplicates(subset=drop_subset, keep='first')
temp1 = all_data.fillna('').groupby(['id','full_name'])['service_name'].apply(lambda x: fix_name(', '.join(x).lower())).sort_index(ascending=False).reset_index()
df_data = all_data_.merge(temp1, left_on=['id', 'full_name'], right_on=['id','full_name'])
df_data.loc[:,'date'] = pd.to_datetime(df_data.loc[:,'date'])

remove_entries = [ 'frig_test', 'sample quotation']
df_data = df_data[df_data.loc[:,'full_name'].isin(remove_entries) == False]
# st.sidebar.header('Name to search:')
# name_search = 'arvin'
# string_search = st.sidebar.text_area('Input ', name_search, height =5)

# st.sidebar.write('Searching for ' + string_search)


st.title('MechaniGo Customer Search')

st.markdown("""
            This app searches for the **name** or **email** you select on the table!\n
            Filter the name/email on the dropdown menu as you hover on the column names. 
            Click on the entry to display data below
            """)
# Reprocess dataframe entries to be displayed
df_temp = df_data.reset_index()[['full_name','email']].drop_duplicates(subset=['full_name','email'], keep='first')
df_temp['full_name'] = df_temp['full_name'].str.title()
df_display = df_temp

gb = GridOptionsBuilder.from_dataframe(df_display)
gb.configure_selection('multiple', use_checkbox=True, groupSelectsChildren="Group checkbox select children") #Enable multi-row selection
gridOptions = gb.build()

data_selection = AgGrid(
    df_display,
    gridOptions=gridOptions,
    data_return_mode='AS_INPUT', 
    update_mode='MODEL_CHANGED', 
    fit_columns_on_grid_load=True,
    theme='blue', #Add theme color to the table
    enable_enterprise_modules=True,
    height=200, 
    reload_data=False
)  
# with st.container():
#     st.write("This is inside the container")
#     AgGrid(df_display, height = 200)    
#data = data_selection['data']
#st.write(len(data_selection['selected_rows'][0]))
selected = data_selection['selected_rows']

if selected:
    if len(selected)==1:
        st.dataframe(search_for_name(selected[0]['full_name']))
        st.write('Found entries: ' + str(len(data_selection['selected_rows']))+
                                      '.'+' Select on the expand button to display data in fullscreen.')
    elif len(selected)>1:
        df_list =[]
        for checked_items in range(len(selected)):
            df_list.append(search_for_name(selected[checked_items]['full_name']))
        st.dataframe(pd.concat(df_list))
        st.write('Checked items: ' + str(len(data_selection['selected_rows']))+
                                      '.'+' Select on the expand button (hover on data table) to display data in fullscreen.')
        st.write('Entries: '+str(len(pd.concat(df_list))))
        df_list = []
else:
    st.write('Please click on an entry in the table to display data.')
