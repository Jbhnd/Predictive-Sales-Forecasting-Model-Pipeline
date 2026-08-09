import io
import joblib
import pandas as pd
import streamlit as st
import os
from charset_normalizer import from_bytes

# Page configuration and title
st.set_page_config(page_title='Businesss Sales Prediction Dashboard', layout='centered')
st.title('Sales Prediction Dashboard')
st.markdown('Upload your raw data file below to generate instant prediction')

st.markdown(
    '''
    <style>
    div.stButton > button.st-emotion-cache-en1taq {background-color: rgba(173, 216, 230, 1);}
    div.stButton > button.st-emotion-cache-en1taq:hover {background-color: rgba(118, 189, 213, 1)}
    </style>
    ''', unsafe_allow_html=True
)



# Page Body
SAMPLE_PATH = './data/sample_data.csv'

# Load sample data
def get_sample_data():
    # Verify file path
    if os.path.exists(SAMPLE_PATH):
        st.info('Loaded built-in sample data.')
        return pd.read_csv(SAMPLE_PATH)
    else:
        st.error(f'System Error: The file {SAMPLE_PATH} was not found')
        return None

# Load pipeline once
# @st.cache_resource
def load_pipeline():
    try:
        return joblib.load('model/pipeline_model.joblib', mmap_mode='r')
    except Exception as e:
        st.error(f'Error loading model file: {e}')
        return None

pipeline = load_pipeline()

# Initialize session states
if 'use_sample' not in st.session_state:
    st.session_state['use_sample'] = False

if 'gen_predictions' not in st.session_state:
    st.session_state['gen_predictions'] = False

def set_state_true(key_name):
    st.session_state[key_name] = True

def set_state_false(key_name):
    st.session_state[key_name] = False

# Provide dataset (Upload or use sample)
st.markdown('### Provide your dataset')
col1, col2 = st.columns([2, 1])

# Provide dataset file upload widget (accepts both csv and excel)
with col1:
    uploaded_file = st.file_uploader(
        'Upload an Excel (.xlsx) or CSV (.csv) file',
        type=['csv', 'xlsx'],
        on_change=set_state_false,
        args=['gen_predictions']
    )

# Use sample file widget
with col2:
    st.markdown('<br>', unsafe_allow_html=True) # Aligns button with file uploader
    st.button('Use Sample Data Instead', use_container_width=True, on_click=set_state_true, args=['use_sample'])

# Read chosen data
raw_df = None
if uploaded_file is not None:
    uploaded_file.seek(0) # ensure file pointer is at beginning
    result = from_bytes(uploaded_file.read()).best() # read file encoding
    uploaded_file.seek(0) # rewind file pointer to beginning
    
    if uploaded_file.name.endswith('.csv'):
        raw_df = pd.read_csv(uploaded_file, encoding=result.encoding) # engine='python'
    else:
        raw_df = pd.read_excel(uploaded_file)
elif st.session_state.use_sample:
    raw_df = get_sample_data()
    st.session_state.use_sample = True


if raw_df is not None:
    try:
        # Display preview of uploaded_file
        st.subheader('Raw Data Preview')
        st.dataframe(raw_df.head(5))
        st.text(f'Rows Loaded: {raw_df.shape[0]}')
        st.button('Generate Predictions', type='primary', on_click=set_state_true, args=['gen_predictions'])
        
        if st.session_state.gen_predictions:
            with st.spinner('Processing data through the pipeline...'):
                # Pass raw dataframe to pipeline
                predictions = pipeline.predict(raw_df)
            
                # Append predictions
                raw_df['Model_Sales_Predictions'] = predictions
                raw_df = raw_df.round(3)
                
                # Display updated preview with predictions included
                st.subheader('Prediction Results')
                st.dataframe(raw_df.head(10))
                st.text(f'Rows Predicted: {raw_df.shape[0]}')
        
                # Convert final data to csv
                csv_data = raw_df.to_csv(index=False).encode('utf-8')

                # Download output
                st.download_button(
                    label='Download Prediction Results as CSV',
                    data=csv_data,
                    file_name='predicted_business_results.csv',
                    mime='text/csv'
                )
    except Exception as e:
        st.error(f'An error occurred while processing the file: {str(e)}')
        st.info('Please verify that your file column names and data closely match the sample feature dataset.')
