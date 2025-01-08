import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import numpy as np


st.set_page_config(page_title='Cancer Prediction')

if 'current_page' not in st.session_state:
    st.session_state['current_page'] = 'main'

# Model building
def build_model(df):
    X = df.iloc[:,:-1] # Using all column except for the last column as X
    Y = df.iloc[:,-1] # Selecting the last column as Y

    # Data splitting
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=(100-split_size)/100)
    
    st.markdown('**1.2. Data splits**')
    st.write('Training set')
    st.info(X_train.shape)
    st.write('Test set')
    st.info(X_test.shape)

    st.markdown('**1.3. Variable details**:')
    st.write('X variable')
    st.info(list(X.columns))
    st.write('Y variable')
    st.info(Y.name)

    rf = RandomForestClassifier(
        n_estimators=parameter_n_estimators,
        random_state=parameter_random_state,
        max_features=parameter_max_features,
        criterion='gini',
        min_samples_split=parameter_min_samples_split,
        min_samples_leaf=parameter_min_samples_leaf,
        bootstrap=parameter_bootstrap,
        oob_score=parameter_oob_score,
        n_jobs=parameter_n_jobs
    )
    rf.fit(X_train, Y_train)

    st.subheader('2. Model Performance')

    # Training performance
    st.markdown('**2.1. Training set**')
    Y_pred_train = rf.predict(X_train)
    st.write('Accuracy:')
    st.info(accuracy_score(Y_train, Y_pred_train))

    st.markdown('**Classification Report (Training set):**')
    report = classification_report(Y_train, Y_pred_train, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    report_df["precision"] = np.round(report_df["precision"], 2)
    report_df["recall"] = np.round(report_df["recall"], 2)
    report_df["f1-score"] = np.round(report_df["f1-score"], 2)
    st.table(report_df)
    # st.text(classification_report(Y_train, Y_pred_train))

    # Test performance
    st.markdown('**2.2. Test set**')
    Y_pred_test = rf.predict(X_test)
    st.write('Accuracy:')
    st.info(accuracy_score(Y_test, Y_pred_test))

    st.markdown('**Classification Report (Test set):**')
    report_test = classification_report(Y_test, Y_pred_test, output_dict=True)
    report_rt = pd.DataFrame(report_test).transpose()
    report_rt["precision"] = np.round(report_rt["precision"], 2)
    report_rt["recall"] = np.round(report_rt["recall"], 2)
    report_rt["f1-score"] = np.round(report_rt["f1-score"], 2)
    st.table(report_rt)
    # st.text(classification_report(Y_test, Y_pred_test))

    st.subheader('3. Model Parameters')
    st.write(rf.get_params())

    # Save the trained model in session state
    st.session_state['model'] = rf
    st.session_state['columns'] = X.columns
    st.success("Model trained successfully!")

    # Display model performance
    accuracy = rf.score(X_test, Y_test)


#---------------------------------#
# Sidebar - Collects user input features into dataframe
with st.sidebar.header('1. Upload your CSV data'):
    uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])

# Sidebar - Specify parameter settings
with st.sidebar.header('2. Set Parameters'):
    split_size = st.sidebar.slider('Data split ratio (% for Training Set)', 10, 90, 80, 5)

with st.sidebar.subheader('2.1. Learning Parameters'):
    parameter_n_estimators = st.sidebar.slider('Number of estimators (n_estimators)', 0, 1000, 100, 100)
    parameter_max_features = st.sidebar.select_slider('Max features (max_features)', options=['sqrt', 'log2', None])
    parameter_min_samples_split = st.sidebar.slider('Minimum number of samples required to split an internal node (min_samples_split)', 1, 10, 2, 1)
    parameter_min_samples_leaf = st.sidebar.slider('Minimum number of samples required to be at a leaf node (min_samples_leaf)', 1, 10, 2, 1)

with st.sidebar.subheader('2.2. General Parameters'):
    parameter_random_state = st.sidebar.slider('Seed number (random_state)', 0, 1000, 42, 1)
    # parameter_criterion = st.sidebar.select_slider('Performance measure (criterion)', options=['mse', 'mae'])
    parameter_bootstrap = st.sidebar.select_slider('Bootstrap samples when building trees (bootstrap)', options=[True, False])
    parameter_oob_score = st.sidebar.select_slider('Whether to use out-of-bag samples to estimate the R^2 on unseen data (oob_score)', options=[False, True])
    parameter_n_jobs = st.sidebar.select_slider('Number of jobs to run in parallel (n_jobs)', options=[1, -1])


# Main panel
#---------------------------------#
if st.session_state['current_page'] == 'main':
    st.write("""
    # Cancer Prediction Application 

    In this implementation, the *RandomForestClassifier()* function is used in this app to build a classification model using the **Random Forest** algorithm.

    Try adjusting the hyperparameters in the left!

    """)
    st.subheader('1. Dataset')

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.markdown('**1.1. Glimpse of dataset**')
        st.write(df)
        # Drop irrelevant columns (example: 'Patient Id')
        if "Patient Id" in df.columns:
            df = df.drop(columns=["Patient Id"])

        # Convert categorical columns to numeric
        for col in df.columns:
            if df[col].dtype == "object":
                st.write(f"Converting non-numeric column: {col}")
                df[col] = pd.Categorical(df[col]).codes  # Label encoding
        build_model(df)


    else:
        st.info('Awaiting for CSV file to be uploaded.')
        if st.button('Press to use Example Dataset'):
            # Load the example dataset from cancer.csv
            df = pd.read_csv(r'C:\Users\sanik\Desktop\Sanika\cancer prediction\Cancer.csv')# Replace with the actual path to cancer.csv
            st.markdown('The Example Dataset (Cancer.csv) is used as the example.')
            st.write(df.head())
            # Drop irrelevant columns (example: 'Patient Id')
            if "Patient Id" in df.columns:
                df = df.drop(columns=["Patient Id"])

            # Convert categorical columns to numeric
            for col in df.columns:
                if df[col].dtype == "object":
                    st.write(f"Converting non-numeric column: {col}")
                    df[col] = pd.Categorical(df[col]).codes  # Label encoding
            build_model(df)

    # Show the button only if the model is trained
    if 'model' in st.session_state:
        if st.button("Go to Prediction Page"):
            st.session_state['current_page'] = 'prediction'
    else:
        st.warning("Train the model first to proceed to the prediction page.")

 # Prediction Page       
elif st.session_state['current_page'] == 'prediction':
    st.title("Make a Prediction")

    if 'model' in st.session_state:
        model = st.session_state['model']
        columns = st.session_state['columns']

        # Input form for prediction
        input_data = []
        for col in columns:
            value = st.number_input(f"Enter {col}:", value=0.0)
            input_data.append(value)

        if st.button("Predict"):
            result = model.predict([input_data])
            if (result==1):
                st.success(f"The predicted class is Low")
            elif (result==2):
                st.success(f"The predicted class is Medium")
            else:
                st.success(f"The predicted class is High")
    else:
        st.warning("Please train the model first!")

    # Add navigation button
    if st.button("Back to Main Page"):
        st.session_state['current_page'] = 'main'
