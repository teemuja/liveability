#utils.py
import streamlit as st
import pandas as pd
import boto3

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
def bucket_handler(bucket_name=None, file_name=None, folder_name=None, operation=None, data_frame=None):
    #session
    session = boto3.session.Session()
    client = session.client('s3',
                            endpoint_url=bucket_url,
                            aws_access_key_id=bucket_key, 
                            aws_secret_access_key=bucket_secret 
                            )
    
    if operation == 'download':
        return download_csv_from_spaces(client, bucket_name, file_name)
    elif operation == 'upload' and data_frame is not None:
        upload_csv_to_spaces(client, bucket_name, file_name, data_frame)
    elif operation == 'list':
        return list_files_from_bucket(client,bucket_name,folder_name)
    else:
        raise ValueError("Invalid operation or missing data for upload")

def download_csv_from_spaces(client, bucket_name, file_name):
    obj = client.get_object(Bucket=bucket_name, Key=file_name)
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
