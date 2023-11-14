import streamlit as st
import pandas as pd
import geopandas as gpd
from shapely.wkt import loads
import json
import plotly.express as px
from pathlib import Path

import utils

st.set_page_config(page_title="LIV", layout="wide")
st.header("Liveability decomposed")
st.markdown("Authors: Tianqi Wang, Teemu Jama, Henrikki Tenkanen @[Aalto GIST Lab](https://gistlab.science)")

@st.cache_data()
def get_gdf(folder_name='ndp/liv', file_name='liveability_4326.csv'):
    df = utils.bucket_handler(operation="download",folder_name=folder_name,file_name=file_name)
    df['geometry'] = df['WKT'].apply(loads)
    gdf = gpd.GeoDataFrame(df,geometry="geometry")
    return gdf

gdf_indexes = get_gdf(folder_name='ndp/liv', file_name='liveability_4326.csv')
gdf_GWR = get_gdf(folder_name='ndp/liv', file_name='GWR.csv')

#selectors
df_to_study = st.radio('Select data to study',options=['Indexes','GWR'])

if df_to_study == "Indexes":
    cols = gdf_indexes.drop(columns=["YKR_ID","WKT","geometry"]).columns.tolist()
    defaults = [cols[0],cols[1]]
    mycols = st.multiselect("Select values to study",options=cols,default=defaults,max_selections=4)
    gdf = gdf_indexes.copy()
    #rename cols
    #json_path = Path(__file__).parent / 'cols_dict.json'
    #with open(json_path) as json_file:
    #    cols_dict = json.load(json_file)
    #gdf.rename(columns=cols_dict,inplace=True)
else:
    cols = gdf_GWR.drop(columns=["YKR_ID","WKT","geometry"]).columns.tolist()
    defaults = [cols[0],cols[1]]
    mycols = st.multiselect("Select values to study",options=cols,default=defaults,max_selections=4)
    gdf = gdf_GWR.copy()

#tabs
tab1,tab2 = st.tabs(['Maps','Distribution'])

with tab1:
    if len(mycols) != 0:
        st.plotly_chart(utils.facet_plots(gdf,cols=mycols,opacity=0.4), use_container_width=True, config = {'displayModeBar': False})
    elif len(mycols) > 3:
        st.warning('Max 3')

with tab2:
    if len(mycols) > 1:
        st.plotly_chart(utils.scatter(gdf,cols=mycols), use_container_width=True, config = {'displayModeBar': False})
    elif len(mycols) == 1:
        st.plotly_chart(px.histogram(gdf,x=mycols), use_container_width=True, config = {'displayModeBar': False})
    else:
        st.empty()
