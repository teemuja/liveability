import streamlit as st
import pandas as pd
import geopandas as gpd
from shapely.wkt import loads

import utils

st.set_page_config(page_title="LIV", layout="wide")
st.header("Liveability decomposed")
st.markdown("Authors: Tianqi Wang, Teemu Jama, Henrikki Tenkanen @[Aalto GIST Lab](https://gistlab.science)")

df = utils.bucket_handler(operation="download",folder_name='ndp/liv', file_name='liveability_4326.csv')
df['geometry'] = df['WKT'].apply(loads)
gdf = gpd.GeoDataFrame(df,geometry="geometry")

#selectors
cols = gdf.drop(columns=["WKT","geometry"]).columns.tolist()
defaults = ['Attractive','Walkabilit']
mycols = st.multiselect("Select values to study",options=cols,default=defaults,max_selections=4)

#tabs
tab1,tab2 = st.tabs(['Map comparisons','Scatter maps'])

with tab1:
    if len(mycols) != 0:
        st.plotly_chart(utils.facet_plots(gdf,cols=mycols), use_container_width=True, config = {'displayModeBar': False})
    elif len(mycols) > 3:
        st.warning('Max 3')

with tab2:
    if len(mycols) > 1:
        st.plotly_chart(utils.scatter(gdf,cols=mycols), use_container_width=True, config = {'displayModeBar': False})
