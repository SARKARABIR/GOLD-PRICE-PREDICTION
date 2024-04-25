import streamlit as st
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load the trained models
linear_regressor = pickle.load(open('linear_regressor.pkl', 'rb'))
random_regressor = pickle.load(open('random_regressor.pkl', 'rb'))
adaboost_regressor = pickle.load(open('adaboost_regressor.pkl', 'rb'))
xgb_regressor = pickle.load(open('xgb_regressor.pkl', 'rb'))
svm_regressor = pickle.load(open('svm_regressor.pkl', 'rb'))
gradient_boost_regressor = pickle.load(open('gradient_boost_regressor.pkl', 'rb'))

# Streamlit app code
def main():
    st.title('Price Prediction App')

    st.sidebar.title('Input Features')
    # Create input fields for user input
    open_input = st.sidebar.number_input('Open', value=100, step=1)
    high_input = st.sidebar.number_input('High', value=100, step=1)
    low_input = st.sidebar.number_input('Low', value=100, step=1)
    # chg_input = st.sidebar.number_input('Chg%', value=0.0, step=0.01, format="%.2f")
    year_input = st.sidebar.number_input('Year', value=2022, step=1)
    month_input = st.sidebar.number_input('Month', value=1, step=1)
    day_input = st.sidebar.number_input('Day', value=1, step=1)

    # Create a dataframe with the user input
    user_input = pd.DataFrame({
        'Open': [open_input],
        'High': [high_input],
        'Low': [low_input],
        'Chg%': 0.00,
        'Year': [year_input],
        'Month': [month_input],
        'Day': [day_input]
    })

    # Dropdown menu to select regression model
    regression_model = st.sidebar.selectbox('Select Regression Model', 
                                            ['Linear Regression', 
                                             'Random Forest Regression', 
                                             'AdaBoost Regression', 
                                             'XgBoost Regression',
                                             'SVM Regression', 
                                             'Gradient Boosting Regression'])

    # Predict the price using the selected regression model

    if regression_model == 'Linear Regression':
        predicted_price = linear_regressor.predict(user_input)/10000
    elif regression_model == 'Random Forest Regression':
        predicted_price = random_regressor.predict(user_input)
    elif regression_model == 'AdaBoost Regression':
        predicted_price = adaboost_regressor.predict(user_input)
    elif regression_model == 'XgBoost Regression':
        predicted_price = xgb_regressor.predict(user_input)
    elif regression_model == 'SVM Regression':
        predicted_price = svm_regressor.predict(user_input)
    elif regression_model == 'Gradient Boosting Regression':
        predicted_price = gradient_boost_regressor.predict(user_input)/10000

    # Display the predicted price
    st.write('Predicted Price:', max(predicted_price))
    #st.write(predicted_price)

if __name__ == '__main__':
    main()
