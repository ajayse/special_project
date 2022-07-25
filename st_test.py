import streamlit as st
import pandas as pd


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
    if n not in name_list:
      name_list.append(n)
  return ' '.join(name_list).strip()

def search_for_name(name):
  df_match = pd.DataFrame(),pd.DataFrame()
  df_data.full_name = df_data.full_name.apply(lambda x: x.lower())
  df_data.email = df_data.email.apply(lambda x: x.lower())
  names = pd.merge(left =df_data.loc[df_data.apply(lambda x: name.lower() in x.full_name, axis=1)],right =df_data.loc[df_data.apply(lambda x: name.lower() in x.email.lower(), axis=1)],
         how ='outer', on =['id','full_name','email','car','appointment_date','mechanic_name','service_name'])
  names.email = names.email.str.lower()
  names.full_name = names.full_name.str.title()
  df_match = names.groupby(['full_name','email','id']).agg(                     
                                                          car =('car', lambda x:', '.join(x.unique())),
                                                          appointment_date =('appointment_date', lambda x: ', '.join([str(i) for i in x.dt.date.unique()])),
                                                          services_availed =('service_name', lambda x: ', '.join(x.unique()))
                                                      ).reset_index()
  df_match.columns = ['full_name','email','transaction_id','car','appointment_date','services_availed']
  df_match = df_match.set_index('full_name')
  df_match.index.name = 'full_name'
  return df_match

all_data = pd.read_csv("http://app.redash.licagroup.ph/api/queries/103/results.csv?api_key=QHb7Vxu8oKMyOVhf4bw7YtWRcuQfzvMS6YBSqgeM", parse_dates=['date','appointment_date','date_confirmed'])
all_data = all_data.rename(columns={'year': 'model_year', 'name':'status'})
all_data = all_data[all_data['status']!='Cancelled']
all_data.loc[:,'full_name'] = all_data.apply(lambda x: fix_name(x['full_name']), axis=1)

cols = ['id', 'date', 'email','full_name','make', 'model', 'model_year', 
        'appointment_date', 'mechanic_name', 'sub_total', 'service_fee', 'total_cost', 
        'date_confirmed', 'status', 'status_of_payment']
drop_subset = ['full_name', 'make', 'model', 'appointment_date']
all_data_ = all_data[cols].drop_duplicates(subset=drop_subset, keep='first')
temp1 = all_data.fillna('').groupby(['id','full_name'])['service_name'].apply(lambda x: fix_name(', '.join(x).lower())).sort_index(ascending=False).reset_index()
df_data = all_data_.merge(temp1, left_on=['id', 'full_name'], right_on=['id','full_name'])
df_data.loc[:,'date'] = pd.to_datetime(df_data.loc[:,'date'])

remove_entries = [ 'frig_test', 'sample quotation']
df_data = df_data[df_data.loc[:,'full_name'].isin(remove_entries) == False]
df_data['car'] = (df_data['make'].str.lower()+ ' '+df_data['model'].str.lower()).str.title()

st.sidebar.header('Name to search:')
name_search = 'arvin'
string_search = st.sidebar.text_area('Input ', name_search, height =5)

st.sidebar.write('Searching for ' + string_search)

results = pd.DataFrame()
results = search_for_name(string_search)

st.write(results)

if string_search == "Enter name here":
    st.text('Waiting for input...')
elif len(results) ==0:
    st.text('No results found.')
