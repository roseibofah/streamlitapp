
#Building an app to predict poverty. We use different ML algorithm but emerged with with the best for the prediction
#Model was trained using the GLSS7 dataset at the household level
# With the individual data been the head of household
# location/region variables are also been used to account for spacial dimension
# Import dependencies
# xxxxxxxxxxxxxxxxxxxx
# zzzzzzzzzzzzzzzzzzzz
# pppppppppppppppppppp

import streamlit as st
import pandas as pd
import os
import pickle
import json
from utils import transform_data
from matplotlib import pyplot as plt
import seaborn as sns

# Set title and write output
st.title(""" 
# Predict Poverty - 🚀
(Note: Value 1=Poor; Vaule 0=Not Poor)""")
st.write('Hit Predict to determine if one is likely to be poor!')

# Load Schema 
with open('schema.json', 'r') as f:
    schema = json.load(f)
# st.write(schema)

# Setup column orders 
column_order_in = list(schema['column_info'].keys())[:-1]
column_order_out = list(schema['transformed_columns']['transformed_columns'])
# st.write(column_order_out)

# SIDEBAR Section
st.sidebar.info('Update these features to predict!')
# Collect the input features 
options = {}
for column, column_properties in schema['column_info'].items(): 
    if column == 'Target_Poor':
        pass 
    # Create numerical sliders 
    elif column_properties['dtype'] == 'int64' or column_properties['dtype'] == 'float64':
        min_val, max_val = column_properties['values']
        data_type = column_properties['dtype']

        feature_mean = (min_val+max_val) / 2
        if data_type == 'int64': 
            feature_mean = int(feature_mean) 

        options[column] = st.sidebar.slider(column, min_val, max_val, value=feature_mean) 
    # Create categorical select boxes
    elif column_properties['dtype'] == 'object': 
        options[column] = st.sidebar.selectbox(column, column_properties['values'])

# Load in model and encoder 
model_path = os.path.join('..','models', 'experiment_1', 'xg.pkl')
with open(model_path, 'rb') as f: 
    model = pickle.load(f)

encoder_path = os.path.join('..','models', 'experiment_1', 'encoder.pkl')
with open(encoder_path, 'rb') as f: 
    onehot = pickle.load(f)

# st.write(options) 

# Make a Prediction
if st.button('Predict'): 
    # Convert options to df 
    scoring_data = pd.Series(options).to_frame().T
    scoring_data = scoring_data[column_order_in]

    # Check datatypes 
    for column, column_properties in schema['column_info'].items(): 
        if column != 'Target_Poor': 
            dtype = column_properties['dtype'] 
            scoring_data[column] = scoring_data[column].astype(dtype)

    # Apply data transformations
    scoring_sample = transform_data(scoring_data, column_order_out, onehot)

    # Render Predictions
    prediction = model.predict(scoring_sample)
    st.write('Predicted Outcome')
    st.write(prediction)
    #st.write('Likely to be poor will have value 1; otherwise 0')
    st.write('Features of Heads, Households, etc used for the ABOVE prediction')
    st.write(options)

# Save Historical Data 
try: 
    historical = pd.Series(options).to_frame().T
    historical['prediction'] = prediction
    if os.path.isfile('historical_data.csv'): 
        historical.to_csv('historical_data.csv', mode='a', header=False, index=False)
    else: 
        historical.to_csv('historical_data.csv', header=True, index=False)
except Exception as e: 
    pass 

st.header('Historical Outcomes')
#if os.path.isfile('historical_data.csv'): 
if st.checkbox('Show historical data dataframe of predictions'):
    hist = pd.read_csv('historical_data.csv')
    st.dataframe(hist)
    fig, ax = plt.subplots()
    sns.countplot(x='prediction', data=hist, ax=ax).set_title('Historical Predictions')
    st.pyplot(fig)
#else: 
#    st.write('No historical data')

st.text_input("Your Comments", key="Any Comment")
