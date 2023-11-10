#utils.py
import streamlit as st
import pandas as pd
import boto3
import numpy as np

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from itertools import combinations

mapbox_key = st.secrets["mapbox"]['MAPBOX_key']
px.set_mapbox_access_token(mapbox_key)
mystyle = st.secrets["mapbox"]['MAPBOX_tiles']

#project bucket keys
bucket_key = st.secrets["bucket"]['BUCKET_accesskey']
bucket_secret = st.secrets["bucket"]['BUCKET_secretkey']
bucket_url = st.secrets["bucket"]['BUCKET_url']
bucket_name = st.secrets["bucket"]['BUCKET_name']

#auth
def check_password():
    """Returns `True` if the user had a correct password."""

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if (
            st.session_state["username"] in st.secrets["passwords"]
            and st.session_state["password"]
            == st.secrets["passwords"][st.session_state["username"]]
        ):
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # don't store username + password
            del st.session_state["username"]
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        # First run, show inputs for username + password.
        st.text_input("Username", on_change=password_entered, key="username")
        st.text_input(
            "Password", type="password", on_change=password_entered, key="password"
        )
        return False
    elif not st.session_state["password_correct"]:
        # Password not correct, show input + error.
        st.text_input("Username", on_change=password_entered, key="username")
        st.text_input(
            "Password", type="password", on_change=password_entered, key="password"
        )
        st.error("😕 User not known or password incorrect")
        return False
    else:
        # Password correct.
        return True
    

#boto3
def bucket_handler(operation, file_name=None, folder_name=None, data_frame=None):
    #session
    session = boto3.session.Session()
    client = session.client('s3',
                            endpoint_url=f"https://{bucket_url}",
                            aws_access_key_id=bucket_key, 
                            aws_secret_access_key=bucket_secret 
                            )
    
    if operation == 'download':
        path_name = f"{folder_name}/{file_name}"
        df = download_csv_from_spaces(client, bucket_name, file_path=path_name)
        return df
    elif operation == 'upload' and data_frame is not None:
        upload_csv_to_spaces(client, bucket_name, file_name, data_frame)
    elif operation == 'list':
        return list_files_from_bucket(client,bucket_name,folder_name)
    else:
        raise ValueError("Invalid operation or missing data for upload")

def download_csv_from_spaces(client, bucket_name, file_path):
    obj = client.get_object(Bucket=bucket_name, Key=file_path)
    df = pd.read_csv(obj['Body'])
    return df

def upload_csv_to_spaces(client, bucket_name, file_name, data_frame):
    csv_buffer = data_frame.to_csv(index=False)
    client.put_object(Bucket=bucket_name, Key=file_name, Body=csv_buffer, ContentType='text/csv')

def list_files_from_bucket(client,bucket_name,folder_name):
    # List objects in the specified folder
    objects = client.list_objects_v2(Bucket=bucket_name, Prefix=folder_name)

    # Initialize a list to hold CSV file names
    csv_files = []

    # Iterate over each object in the specified folder
    for obj in objects.get('Contents', []):
        file_name = obj['Key']
        # Check if the file is a CSV
        if file_name.endswith('.csv'):
            csv_files.append(file_name)

    return csv_files


#colors
bin_colors = {
    'bottom':'antiquewhite',
    'low':'burlywood',
    'high':'olive',
    'top':'darkgreen'
}

def plot_quantiles(gdf,col=None):
    #quantile bins
    median = gdf[col].median()
    min1 = gdf[col].quantile(0.25)
    max1 = gdf[col].quantile(0.75)
    bins = [-np.inf, min1, median, max1, np.inf]
    dis_labels = ['bottom', 'low', 'high', 'top']
    gdf['color_bin'] = pd.cut(x=gdf[col], bins=bins, labels=dis_labels)
    
    lat = gdf.unary_union.centroid.y
    lon = gdf.unary_union.centroid.x
    fig = px.choropleth_mapbox(gdf,
                                geojson=gdf.geometry,
                                locations=gdf.index,
                                color="color_bin",
                                hover_name='YKR_ID',
                                center={"lat": lat, "lon": lon},
                                mapbox_style=mystyle,
                                color_discrete_map=bin_colors,
                                category_orders={'color_bin':['bottom','low','high','top']},
                                zoom=10,
                                opacity=0.5,
                                width=1200,
                                height=900
                                )
    return fig


def facet_plots(gdf,cols):
    #cols
    columns_to_plot = cols

    # Define the number of columns for the subplots
    num_cols = len(columns_to_plot)

    # Create a subplot for each column
    fig = make_subplots(rows=1, cols=num_cols, 
                    subplot_titles=columns_to_plot,
                    specs=[[{"type": "choroplethmapbox"} for _ in range(num_cols)]],
                    horizontal_spacing=0.02)
    
    # Loop through each column to create a choropleth map
    for i, column in enumerate(columns_to_plot):

        # Calculate quartiles
        q1, q2, q3 = gdf[column].quantile([0.25, 0.5, 0.75])

        # Define the maximum value for normalization
        max_value = gdf[column].max()

        # Create a custom color scale for this column based on quartiles
        custom_color_scale = [
            [0, bin_colors['bottom']],
            [q1 / max_value, bin_colors['low']],
            [q2 / max_value, bin_colors['high']],
            [q3 / max_value, bin_colors['top']],
            [1.0, bin_colors['top']]
        ]

        # Create the choropleth trace
        trace = go.Choroplethmapbox(
            geojson=gdf.geometry.__geo_interface__, 
            locations=gdf.index, 
            z=gdf[column],
            colorscale=custom_color_scale,
            zmin=gdf[column].min(),
            zmax=gdf[column].max(),
            showscale=False,
            #colorbar=dict(thickness=20, tickvals=[q1,q2,q3]),
            subplot=f'mapbox{i+1}'
        )

        # Add the trace to the subplot
        fig.add_trace(trace, row=1, col=i+1)

    # Set the mapbox configuration for each subplot
    for i in range(len(columns_to_plot)):
        fig.update_layout(
            **{f'mapbox{i+1}': dict(
                center=dict(lat=gdf.unary_union.centroid.y, lon=gdf.unary_union.centroid.x), # Adjust the center as needed
                style='carto-positron',
                zoom=9
            )}
        )

    # Update the layout for the entire figure
    fig.update_layout(
        height=700,
        width=700 * num_cols,
        margin=dict(l=10, r=10, t=50, b=10)
    )
    
    return fig



def corr_heat(df,cols):
        # Calculate the correlation matrix
        correlation_matrix = df[cols].corr()

        # Create a heatmap using Plotly
        fig = px.imshow(correlation_matrix, 
                        text_auto=True,  # Display correlation values in the cells
                        labels=dict(x="Variable", y="Variable", color="Correlation"),
                        x=correlation_matrix.columns,
                        y=correlation_matrix.columns)

        # Update layout for clarity
        fig.update_layout(title="Correlation Heatmap", 
                        xaxis_title="Variable", 
                        yaxis_title="Variable",
                        coloraxis_showscale=False)

        return fig

def scatter(df,cols):
    # Determine the number of rows and columns for the subplots
    num_plots = len(cols) * (len(cols) - 1) // 2
    num_rows = int(num_plots**0.5)
    num_cols = (num_plots + num_rows - 1) // num_rows  # Ensure enough columns

    # Create a subplot figure
    fig = make_subplots(rows=num_rows, cols=num_cols, subplot_titles=[f'{x[0]} vs {x[1]}' for x in combinations(cols, 2)])

    # Add a scatter plot to each subplot cell
    for i, (col1, col2) in enumerate(combinations(cols, 2), start=1):
        fig.add_trace(
            go.Scatter(x=df[col1], y=df[col2], mode='markers', name=f'{col1} vs {col2}'),
            row=(i - 1) // num_cols + 1,
            col=(i - 1) % num_cols + 1
        )

    # Update layout
    fig.update_layout(height=600, width=1200, title_text="Scatter plot")
    return fig