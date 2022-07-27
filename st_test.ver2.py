import streamlit as st
import pandas as pd
import math


st.title('MechaniGo Customer Search')

st.markdown("""
            This app searches for the **name** or **email** you enter on the sidebar!
            """)
            
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
    if n not in name_list and n!='':
      name_list.append(n)
  return ' '.join(name_list).strip()


def search_for_name(name):
  df_match = pd.DataFrame(),pd.DataFrame()
  df_data.full_name = df_data.full_name.apply(lambda x: x.lower())
  df_data.email = df_data.email.apply(lambda x: x.lower())
  left_df =df_data.loc[df_data.apply(lambda x: name.lower() in x.full_name, axis=1)]
  right_df =df_data.loc[df_data.apply(lambda x: name.lower() in x.email.lower(), axis=1)]
  if len(left_df) + len(right_df)>0:
      names = pd.merge(left_df,right_df,
             how ='outer', on =['id','full_name','email','brand', 'model/year','appointment_date','mechanic_name','service_name','customer_id','fuel_type','transmission','plate_number','phone','address','mileage'])
      names.email = names.email.str.lower()
      names.full_name = names.full_name.str.title()
  
      df_match = names.groupby(['full_name','email','id']).agg(
                                                              customer_id=('customer_id', lambda x:' ' if x.empty else x),
                                                              brand=('brand', lambda x:', '.join(x.unique())),
                                                              model =('model/year', lambda x:', '.join(x.unique())),
                                                              fuel_type =('fuel_type', lambda x:' ' if x.empty else x),
                                                              transmission =('transmission', lambda x:' ' if x.empty else x),
                                                              plate_number =('plate_number', lambda x:' ' if x.empty else x.str.upper()),
                                                              phone=('phone', lambda x:'' if x.empty else ', '.join(x.unique())),
                                                              address=('address', lambda x:'' if x.empty else ', '.join(x.unique())),
                                                              mileage=('mileage', lambda x:'' if math.isnan(x) else str(x)),
                                                              appointment_date =('appointment_date', lambda x: ', '.join([str(i) for i in x.dt.date.unique()])),
                                                              services_availed =('service_name', lambda x: ', '.join(x.unique()))
                                                          ).reset_index()
      df_match.columns = ['full_name','email','transaction_id','customer_id', 'brand', 'model/year','fuel_type','transmission','plate_number','contact_no','address','mileage','appointment_date','services_availed']
      df_match = df_match.set_index('customer_id')
      df_match.index.name = 'customer_id'
      df_match = df_match[['full_name','brand','model/year','fuel_type','transmission','plate_number','contact_no','address','mileage','appointment_date','services_availed','email']]
  elif len(left_df) + len(right_df)==0:
      df_match = pd.DataFrame()
  return df_match


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

st.sidebar.header('Name to search:')
name_search = 'arvin'
string_search = st.sidebar.text_area('Input ', name_search, height =5)

st.sidebar.write('Searching for ' + string_search)

results = pd.DataFrame()
results = search_for_name(fix_name(string_search))

if string_search == "Enter name here":
    st.text('Waiting for input...')
elif len(results) ==0:
    st.text('No results found.')
elif len(results) >0:
    st.dataframe(results)
