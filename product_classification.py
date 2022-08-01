# -*- coding: utf-8 -*-
"""
Created on Thu Jul 28 18:59:13 2022

@author: Arvin Jay
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
pd.options.mode.chained_assignment = None

st.set_page_config(
    page_title='GoParts Product Classification (ABC-XYZ)',
    initial_sidebar_state='expanded',
    menu_items = {
        'About':"# Thanks for using the app!\n For inquiries, send an email to Arvin Escolano at arvinjayescolano.licagroup@gmail.com."
        }
    )
began = 0
segment ='brand'

all_data = pd.read_csv("http://app.redash.licagroup.ph/api/queries/118/results.csv?api_key=nVfPq3pxbOF6uSWOlCI8HQSRmgMb34OD6tWvrapY", parse_dates=['created_at'])
all_data = all_data.drop(all_data[all_data.price < 0].index)
all_data['garage_type'].loc[all_data['garage_type'] == 'nonlica_delearship'] = 'nonlica_dealership'
all_data['garage_type'].loc[all_data['garage_type'] == 'Inactive'] = 'B2C'
df = all_data[['id', 'GarageId', 'garage_type', 'created_at', 'product cost','price','quantity','Income','total_price_no_shipping','product_id','product_desc','category_name','brand']].copy()
df.columns = ['id', 'garage_id', 'garage_type', 'date', 'cost','price','quantity','consumption_value','total_price_no_shipping','product_id','product_desc','category_name','brand']
df['brand_category'] = df['brand'] +'_'+df['category_name']
df = df.loc[df['consumption_value']>0]

def get_data(df_input,segment,date_interval):
  df_xyz = pd.DataFrame()
  df_data = df_input.groupby([pd.Grouper(freq = date_interval, key='date'), segment])['quantity'].sum().fillna(0).unstack().reset_index()
  df_data['date'] = df_data['date'].apply(lambda x: x.date())
  df_data = df_data.set_index('date')
  df_d = df_data.describe().loc[['count','mean','std','min','max']]
  df_temp =df_data.copy()
  df_xyz = round((df_temp.std()/df_temp.mean()),2).to_frame('cv').sort_values(by='cv', ascending = True)
  df_xyz['XYZ'] = df_xyz['cv'].apply(get_xyz)
  df_xyz.index.name = segment

  df_data_abc = pd.DataFrame()
  df_data_abc[['cvalue','quantity']] = df_input.groupby(segment).agg(
                                                        cvalue =('consumption_value', lambda x: x.sum()),
                                                        quantity =('quantity', lambda x: x.sum())
                                                    )
  df_abc = df_data_abc.sort_values(by='cvalue', ascending=True)
  df_abc['cumulative_pct'] = df_abc['cvalue'].cumsum()/df_abc['cvalue'].sum()
  df_abc['ABC'] = df_abc['cumulative_pct'].apply(get_abc)
  df_abc = df_abc.sort_values(by='ABC',ascending = True)
  df_abc.index.name = segment

  df_summary = pd.DataFrame()
  df_summary = pd.concat([df_xyz, df_abc], axis=1)
  df_summary = df_summary[['ABC', 'XYZ', 'cvalue','quantity', 'cv']].sort_values(by='cvalue', ascending=False)
  df_summary['cv'] =df_summary['cv'].fillna(9.99)
  df_summary['XYZ'] =df_summary['XYZ'].fillna('Z')
  df_summary['cumulative_pct'] = round((100*df_summary['cvalue']/df_summary['cvalue'].sum()).cumsum(),2)
  df_summary = pd.merge(left =df_summary,right=df_d.transpose(),on=segment)
  df_summary['class']= df_summary['ABC']+df_summary['XYZ']
  
  
  return df_data, df_summary

def get_abc(i_rank):
  if i_rank <=0.05:
    return 'C'
  elif i_rank <=0.2:
    return 'B'
  elif i_rank <= 1.01:
    return 'A'
  else:
    return 'C'

def get_xyz(cv):
  if cv <= 0.6:
    return 'X'
  elif cv <= 1.0:
    return 'Y'
  elif cv >1:
    return 'Z'

def find_class(item):
  return df_summary.loc[item]['class']

def find_xyz(item):
  return df_summary.loc[item]['XYZ']

def find_color(item):
  if find_xyz(item) == 'X':
    return '#000080'
  elif find_xyz(item) == 'Y':
    return '#008080'
  elif find_xyz(item) =='Z':
    return '#800080'

def plot_pareto():
  a_items = df_summary.loc[df_summary['ABC'] == 'A']
  b_items = df_summary.loc[df_summary['ABC'] == 'B']
  c_items = df_summary.loc[df_summary['ABC'] == 'C']
  x_items = df_summary.loc[df_summary['XYZ'] == 'X']
  y_items = df_summary.loc[df_summary['XYZ'] == 'Y']
  z_items = df_summary.loc[df_summary['XYZ'] == 'Z']
  sum = df_summary['cvalue'].sum()


  fig = make_subplots(specs=[[{"secondary_y": True}]])


  fig.add_trace(
      go.Bar(x=z_items.index, y=z_items['cvalue'],
            width=1,
            marker_color='#800080',
            name="Class Z"),
      secondary_y=False,
  )

  fig.add_trace(
      go.Bar(x=y_items.index, y=y_items['cvalue'],
            width=1,
            marker_color ='#008080',
            name="Class Y"),
      secondary_y=False,
  )


  fig.add_trace(
      go.Bar(x=x_items.index, y=x_items['cvalue'],
            width=1,
            marker_color ='#000080',
            name="Class X"),
      secondary_y=False,
  )

  fig.add_trace(
      go.Scatter( x=df_summary.index, y=df_summary['cumulative_pct'], name='Cumulative % Total',
      mode = 'lines+markers', marker =dict(
          size = 5,
          color = df_summary['quantity'],
          colorscale = 'phase',
          showscale =False
      ),
      hoverinfo = 'all',
      text = df_summary['cvalue'].apply(lambda x: str( round(100*x/sum ,2)  )) + '% Total Profit '),
      secondary_y=True,
  )

  fig.add_vrect(
      x0=-1, x1=len(a_items)-0.5,
      fillcolor = 'LightGreen',
      opacity = 0.5,
      line_width = 0,
      layer = 'below',
      secondary_y=True,
      name = 'A',
  )
  fig.add_vrect(
      x0=len(a_items)-.5, x1=len(a_items)+len(b_items)-1.5,
      fillcolor = 'Yellow',
      opacity = 0.25,
      line_width = 0,
      layer = 'below',
      secondary_y=True,
      name = 'B',
  )
  fig.add_vrect(
      x0=len(a_items)+len(b_items)-1.5, x1=len(a_items)+len(b_items)+len(c_items)-.5,
      fillcolor = 'LightSalmon',
      opacity = 0.5,
      line_width = 0,
      layer = 'below',
      secondary_y=True,
      name = 'C',
  )
  fig.update_xaxes(categoryorder='array', categoryarray = df_summary.index)

  fig.update_layout(
      title_text="Pareto Chart", plot_bgcolor='white'
  )


  # Set x-axis title
  fig.update_xaxes(title_text=f"<b>{segment}</b>", showline=True, linewidth=1, linecolor='black', mirror=True)

  # Set y-axes titles
  fig.update_yaxes(title_text="<b>Income </b> (PHP)",showgrid= False, secondary_y=False)
  fig.update_yaxes(title_text="<b>Cumulative Percentage</b> (%)", showline=True, linewidth=1, linecolor='black', mirror=True,secondary_y=True)


  st.plotly_chart(fig)
  
def stacked_bar():
  fig_element = []
  for item in df_data.columns:
    fig_element.append(
        go.Bar(name=f"{item}"+"("+find_class(item)+")", 
                x=df_data.index, 
                y=df_data[item], 
                marker=dict(
                    colorscale="Cividis"
                    )
                #text=df_data[item],
                #textposition = 'inside'
                ))

  fig = go.Figure(
      data=fig_element
  )
  fig.add_trace(
      go.Scatter(name='total', 
              mode = 'markers',
              marker =dict(size = 0.01),
              x=df_data.index, 
              y=df_data.sum(axis=1), 
              text=df_data.sum(axis=1),
              #colorscale="Cividis"
              )
  )
  # Change the bar mode
  fig.update_layout(barmode='stack')

  fig.update_layout(
      title_text="Demand Chart", 
      plot_bgcolor='white',
      xaxis = dict(
          title_text="<b>Date </b>", 
          linewidth = 0.1,
          mirror = True,
          showline=True,
          ticks = 'inside',
      ),
      yaxis = dict(
          title_text="<b>Quantity </b>",
          gridcolor = 'LightGrey',
          linewidth = 0.1,
          mirror = True,
          showline= True,
          ticks = 'inside',
      )
  )

  fig.update_xaxes(showline=True, linewidth=1, linecolor='black', mirror=True)
  fig.update_yaxes(showline=True, linewidth=1, linecolor='black', mirror=True)
  st.plotly_chart(fig)
  
def single_demand(item = df[segment].unique()[0]):
  fig = go.Figure()

  # Add traces

  fig.add_trace(go.Bar(x=df_data.index, 
        y=df_data[item],
        marker_color = find_color(item),
        hovertext = find_class(item),
        textposition = 'outside',
        text = df_data[item],
        name = f"{item}" + "("+find_class(item)+")")
  )
  fig.add_trace(go.Scatter(
      x=df_data.index,
      y=[df_data[item].mean(axis=0)]*len(df_data.index),
      mode = 'lines',
      line =dict(
          dash='dash',
          color = 'red'
      ),
      name= 'Average quantity sold per time interval'
  )
      
  )
  fig.update_layout(
      title_text="Demand Chart for "+f"<b>{item}</b>", 
      plot_bgcolor='white',
      xaxis = dict(
          title_text="<b>Date </b>", 
          linewidth = 0.1,
          mirror = True,
          showline=True,
          ticks = 'inside',
      ),
      yaxis = dict(
          title_text="<b>Quantity </b>",
          gridcolor = 'LightGrey',
          linewidth = 0.1,
          mirror = True,
          showline= True,
          ticks = 'inside',
      )
  )

  fig.update_xaxes(showline=True, linewidth=1, linecolor='black', mirror=True)
  fig.update_yaxes(showline=True, linewidth=1, linecolor='black', mirror=True)

  st.plotly_chart(fig)
  info_summary =dict(
      classification =find_class(item),    
      consumption_value_PHP =df_summary.loc[item]['cvalue'],
      variation_coefficient =df_summary.loc[item]['cv'],
      average_quantity = df_summary.loc[item]['mean'],
      total_quantity =df_summary.loc[item]['quantity']
  )
  results = pd.DataFrame([info_summary],index =[item]).transpose()
  results = results.dtypes.astype(str)
  st.write(results)
  
def single_avgdemand(item = df[segment].unique()[0]):
  container = st.container()

  df_temp = df_input.copy()
  df_temp =df_temp.loc[df_temp[segment]==item][['date','quantity']]
  df_forplot = df_temp.groupby(pd.Grouper(freq=date_interval, key='date'))['quantity'].describe().fillna(0)

  fig = go.Figure()
  # Add traces
  fig.add_trace(go.Bar(
      x=df_forplot.index,
      y=df_forplot['mean'],
      marker_color = find_color(item),
      error_y = dict(
          type='data',
          array = df_forplot['std'],
          visible=True
      ),
      hovertext = df_forplot.apply(lambda x: 'Total: '+str((x['count']*x['mean']).astype(int)) +'<br>'+
                                  'Classification: '+str(find_class(item)), axis=1),
      text = df_forplot['mean'].round(2),
      textposition='outside',
      name= 'Average quantity sold per time interval'
  )
      
  )
  fig.update_layout(
      title_text="Demand Chart for "+f"<b>{item}</b>("+find_class(item)+")", 
      plot_bgcolor='white',
      xaxis = dict(
          title_text="<b>Date </b>", 
          linewidth = 0.1,
          mirror = True,
          showline=True,
          ticks = 'inside',
      ),
      yaxis = dict(
          title_text= f"<b>Average {time_interval} Demand </b> (units)",
          gridcolor = 'LightGrey',
          linewidth = 0.1,
          mirror = True,
          showline= True,
          ticks = 'inside',
      )
  )

  fig.update_xaxes(showline=True, linewidth=1, linecolor='black', mirror=True)
  fig.update_yaxes(showline=True, linewidth=1, linecolor='black', range =[0,df_forplot['mean'].max()+df_forplot['std'].max()],fixedrange=True, mirror=True)

  container.plotly_chart(fig)
  #summary df
  info_summary ={
      'item':item,
      'classification' :find_class(item),    
      'consumption_value_PHP' :df_summary.loc[item]['cvalue'],
      'variation_coefficient' :df_summary.loc[item]['cv'],
      'average_quantity' : df_summary.loc[item]['mean'],
      'total_quantity' :df_summary.loc[item]['quantity']
  }
  results = pd.DataFrame(info_summary,index =[0])
  results =results.set_index('item')
  container.write(results)

def convert_df(df):
    return df.to_csv().encode('utf-8')

st.sidebar.header('Parameters for Analysis')
st.sidebar.text('Select the parameters that would\nbe taken into account during the\nanalysis.')
segment_i = st.sidebar.selectbox(
    label = 'Product Segmentation:',
    options =('','SKU', 'Product category', 'Brand','Category per Brand'))
segment_dict = {'SKU': 'product_desc',
    'Product category' : 'category_name',
    'Brand' : 'brand',
    'Category per Brand' :'brand_category'
    }
time_interval = st.sidebar.radio(
     "Select date interval:",
     ('Monthly', 'Weekly', 'Daily'))

if time_interval == 'Daily':
     date_interval = 'D'
elif time_interval == 'Weekly':
     date_interval = 'W'
elif time_interval == 'Monthly':
     date_interval = 'M'

date_start = st.sidebar.date_input(
     label = "Select start of considered dates",
     value = df['date'].min().date(),
     min_value=df['date'].min().date(),
     max_value=df['date'].max().date())
date_end = st.sidebar.date_input(
     label = "Select end of considered dates",
     value = df['date'].mean().date(),
     min_value=df['date'].min().date(),
     max_value=df['date'].max().date())

st.sidebar.write("Selected: "+', '.join([ time_interval, segment_i]))

if  segment_i != '':
    st.sidebar.button('Refresh')
    began = 1
else:
    st.sidebar.write('Select Product segmentation')

    
if began == 0:  
    st.title('GoParts Product Classification')
    
    st.markdown("""
                This app classifies the products of GoParts based on **ABC**-**XYZ** classification. 
                Relevant data are also presented such as the demand per indicated time interval. 
                Kindly select the _parameters_ in the sidebar to begin analysis.
                """)    
elif (date_start>date_end):
    began ==0
    st.title('GoParts Product Classification')
    
    st.markdown("""
                This app classifies the products of GoParts based on **ABC**-**XYZ** classification. 
                Relevant data are also presented such as the demand per indicated time interval. 
                Kindly select the _parameters_ in the sidebar to begin analysis.
                """)   
    st.title("Please indicate a valid date interval.")

elif began ==1 and (date_start<date_end):
    st.title("Inventory Classification for GoParts by " + time_interval+" Demand per "+segment_i+" based on **ABC**-**XYZ** Classification")
    segment = segment_dict[segment_i]
    df_input = df.loc[df['date'] > np.datetime64(date_start)].loc[df['date']< np.datetime64(date_end)]
    df_data, df_summary = get_data(df_input,segment,date_interval)
    
    intro_text = st.container()
    pareto_text = st.container()
    plot_pareto()
    sbar_text = st.container()
    stacked_bar()
    
    st.header('Demand per '+segment_i+":")

    item_i = st.selectbox(
        label = 'Select '+segment_i+' to view its '+time_interval.lower()+' demand:',
        options =tuple(df_summary.index)
        )
    single_avgdemand(item =item_i)
    avgdemand_text = st.container()
    

    
    
        
    
    df_final=df_summary.reset_index().groupby('class').agg(
                                segment_count=(segment, lambda x: x.nunique()),
                                total_demand=('quantity', lambda x: int(x.sum())),
                                avg_demand=('quantity', lambda x:int(x.mean())),
                                total_profit=('cvalue', lambda x: round(x.sum(),2)),
                                members = (segment,lambda x: ', '.join(x.unique())),
        )
    results_text = st.container()
    st.dataframe(df_final)
    
    
    with st.container():
        classifications_text = st.container()
        st.text('All classifications:')
        ax, ay, az = st.columns(3)
        with ax:
            ax_0 = st.checkbox('Show AX data')
            st.markdown("""
                        <p style="text-align: center;"><b>AX</b></p>
                        High profit value, consistent orders, easily forecasted, recommended to be stocked.""", unsafe_allow_html=True)
        with ay:
            ay_0 = st.checkbox('Show AY data')
            st.markdown("""
                        <p style="text-align: center;"><b>AY</b></p>
                        High profit value, variable orders or demand, recommended to be stocked.""", unsafe_allow_html=True) 
        with az:
            az_0 = st.checkbox('Show AZ data')
            st.markdown("""
                        <p style="text-align: center;"><b>AZ</b></p>
                        High profit value, highly inconsistent demand, difficult to be forecasted.""", unsafe_allow_html=True) 
        bx, by, bz = st.columns(3)
        with bx:
            st.markdown(' --- ')
            bx_0 = st.checkbox('Show BX data')
            st.markdown("""
                        <p style="text-align: center;"><b>BX</b></p>
                        Medium profit value, consistent orders, easily forecasted, recommended to be stocked.""", unsafe_allow_html=True)
        with by:
            st.markdown(' --- ')
            by_0 = st.checkbox('Show BY data')
            st.markdown("""
                        <p style="text-align: center;"><b>BY</b></p>
                        Medium profit value, variable orders or demand.""", unsafe_allow_html=True)  
        with bz:
            st.markdown(' --- ')
            bz_0 = st.checkbox('Show BZ data')
            st.markdown("""
                        <p style="text-align: center;"><b>BZ</b></p>
                        Medium profit value, highly inconsistent demand, difficult to forecast.""", unsafe_allow_html=True)  
        cx, cy, cz = st.columns(3)
        with cx:
            st.markdown(' --- ')
            cx_0 = st.checkbox('Show CX data')
            st.markdown("""
                        <p style="text-align: center;"><b>CX</b></p>
                        Low profit value, consistent orders, easily forecasted.""", unsafe_allow_html=True)  
        with cy:
            st.markdown(' --- ')
            cy_0 = st.checkbox('Show CY data')
            st.markdown("""
                        <p style="text-align: center;"><b>CY</b></p>
                        Low profit value, variable orders or demand.""", unsafe_allow_html=True)   
        with cz:
            st.markdown(' --- ')
            cz_0 = st.checkbox('Show CZ data')
            st.markdown("""
                        <p style="text-align: center;"><b>CZ</b></p>
                        Low profit value, highly inconsistent demand.""", unsafe_allow_html=True)   
        data_filter = np.array([ax_0,ay_0,az_0,bx_0,by_0,bz_0,cx_0,cy_0,cz_0])
        class_elements = np.array(['AX','AY','AZ','BX','BY','BZ','CX','CY','CZ'])
        if sum(data_filter)>0:
            df_show = df_summary[df_summary['class'].isin(class_elements[data_filter])][['class','cvalue','quantity']]
            df_show.columns = ['Classification','Profit (PHP)','Quantity Sold (units)']
            if len(df_show) >0:
                st.dataframe(df_show)
            else:
                st.write('No items are included.')
        else:
            st.write('Include at least 1 classification above.')
    
    
    with st.expander('All data:'):
        df_all_show = df_summary[['class','cvalue', 'quantity', 'cv']].copy()
        df_all_show.columns = ['Classification', 'Profit (PHP)', 'Quantity Sold (units)', 'Coefficient of Variation']
        st.dataframe(df_all_show)
     
    st.header("Download data as .csv files")
    st.text("Click on the download button to save the supporting data for the analysis.")
    save1, save2, save3 = st.columns(3)
    with save1:
        st.download_button(
            label = "Download results",
            data = convert_df(df_final),
            file_name="classification_results_summary.csv",
            mime ='text/csv'
            )  
        st.caption("Information on segment count, total demand, average demand, total profit, and members per ABC-XYZ classification")
    with save2:
        st.download_button(
            label = "Download classification data",
            data = convert_df(df_all_show),
            file_name="abc_xyz_results.csv",
            mime ='text/csv'
            )  
        st.caption("Information on classification, profit, quantity, and coefficient of variation per "+segment_i.lower())
    with save3:
        st.download_button(
            label = "Download demand data",
            data = convert_df(df_data.fillna(0)),
            file_name="demand_summary.csv",
            mime ='text/csv'
            )  
        st.caption("Information on the "+ time_interval.lower() +" demand per "+ segment_i.lower()+" with respect to the selected date interval.")
        

        
    intro_text.write("**Introduction**")
    intro_text.write("The ABC-XYZ is a paired classification for inventory itmes that would describe the importance of the "+segment_i+" to the business and and its variablity in demand. The **ABC classification** classifies a "+segment_i+" based on its contribution in profit. On the other hand, the **XYZ-classification** describes the variability in demand with respect to the coefficient of variation on its "+time_interval.lower()+" demand. The analysis below reflects the relevant information that would potentially help us in managing the inventory.")
    pareto_text.header('Pareto Analysis')
    pareto_text.write(
        "The pareto chart for the analysis is shown below. The bar plot corresponds to the consumption value (profit) in PHP represents the contribution of each "+segment_i.lower()+". On the other hand, the line plot shows the cumulattive percentage of the consumption value as we rank the decreasing consumption value of each segment. The most valueable items of the segment is constituted by the items that compose 80% of the total income, which we classify as A (highlighted as green)."+ 
        "Now, the other componenets that constitute up to 95% of the income would be classified as B (highlighted as yellow), while the rest are classified as C (highlighted as red)."
                      )
    sbar_text.header('Overall Demand')
    sbar_text.write(
        "The demand for each "+segment_i.lower()+" is shown below."
        )
    avgdemand_text.write('The average '+time_interval+' demand for '+item_i+' is shown above. We can see that the **XYZ** classification would tell how the demand for the respective '+segment_i+' would vary across time. Moreover, the errorbars would indicate its variation per order at any given time over the indicated time interval. This would provide us with some insight on how we can manage the stock and lay out expectations on what would be its demand.')
    results_text.header("Results")
    results_text.text('The summary of relevant parameters per each classifcation is shown below. The count, total demand, average demand, total profit, and the respective '+segment_i+' that falls into each product classification are summarized in the given table. The descriptions for the respective classifications are shown further below')
    
