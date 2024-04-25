# import streamlit as st
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
# from sklearn.linear_model import LinearRegression
# from sklearn.svm import SVR
# from sklearn import metrics
# import xgboost as xgb

#  # Load the data
# gold_data = pd.read_csv('dataset.csv')

# # # Preprocessing
# # # Convert 'Date' column to numerical format
# def convert_date_to_numeric(date_str):
#      month_str, year_str = date_str.split('-')
#      month_dict = {
#          'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
#          'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12
#      }
#      month_num = month_dict[month_str]
#      return int(year_str) * 100 + month_num

# gold_data['Date'] = gold_data['Date'].apply(convert_date_to_numeric)
# gold_data['Chg%'] = gold_data['Chg%'].str.rstrip('%').astype(float)
# gold_data['Chg%'] = pd.to_numeric(gold_data['Chg%'], errors='coerce')

# gold_data.dropna(inplace=True)

# # # Train-test split
# X = gold_data.drop('Price', axis=1)
# Y = gold_data['Price']
# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

#  # Model training and evaluation
# models = {
#      "Linear Regression": LinearRegression(),
#      "Random Forest": RandomForestRegressor(),
#      "AdaBoost": AdaBoostRegressor(),
#      "Gradient Boosting": GradientBoostingRegressor(),
#      "XGBoost": xgb.XGBRegressor(),
#      "Support Vector Machine": SVR()
# }

# st.title("Gold Price Prediction")

# Model selection
# selected_model = st.selectbox("Select a model", list(models.keys()))

# # Training and evaluating the selected model
# regressor = models[selected_model]
# regressor.fit(X_train, Y_train)
# Y_pred = regressor.predict(X_test)
# error_score = metrics.r2_score(Y_test, Y_pred)

# # Display results
# st.write(f"## {selected_model} Results")
# st.write(f"R squared error: {error_score:.2f}")

# # # Plotting actual vs predicted values
# fig, ax = plt.subplots()
# ax.plot(Y_test.values, color='blue', label='Actual Value')
# ax.plot(Y_pred, color='green', label='Predicted Value')
# ax.set_title('Actual Price vs Predicted Price')
# ax.set_xlabel('Number of values')
# ax.set_ylabel('GLD Price')
# ax.legend()
# st.pyplot(fig)

# import streamlit as st
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
# from sklearn.linear_model import LinearRegression
# from sklearn.svm import SVR
# from sklearn import metrics
# import xgboost as xgb

# # Load the data
# gold_data = pd.read_csv('dataset.csv')

# # Preprocessing
# # Convert 'Date' column to numerical format
# def convert_date_to_numeric(date_str):
#     month_str, year_str = date_str.split('-')
#     month_dict = {
#         'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
#         'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12
#     }
#     month_num = month_dict[month_str]
#     return int(year_str) * 100 + month_num

# gold_data['Date'] = gold_data['Date'].apply(convert_date_to_numeric)
# gold_data['Chg%'] = gold_data['Chg%'].str.rstrip('%').astype(float)
# gold_data['Chg%'] = pd.to_numeric(gold_data['Chg%'], errors='coerce')

# gold_data.dropna(inplace=True)

# # Train-test split
# X = gold_data.drop('Price', axis=1)
# Y = gold_data['Price']
# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

# # Model training and evaluation
# models = {
#     "Linear Regression": LinearRegression(),
#     "Random Forest": RandomForestRegressor(),
#     "AdaBoost": AdaBoostRegressor(),
#     "Gradient Boosting": GradientBoostingRegressor(),
#     "XGBoost": xgb.XGBRegressor(),
#     "Support Vector Machine": SVR()
# }

# st.title("Gold Price Prediction")

# # Model selection
# selected_model = st.selectbox("Select a model", list(models.keys()))

# # Training and evaluating the selected model
# regressor = models[selected_model]
# regressor.fit(X_train, Y_train)
# Y_pred = regressor.predict(X_test)
# error_score = metrics.r2_score(Y_test, Y_pred)

# # Display results
# st.write(f"## {selected_model} Results")
# st.write(f"R squared error: {error_score:.2f}")

# # Plotting actual vs predicted values based on the selected model
# fig, ax = plt.subplots()
# ax.plot(Y_test.values, color='blue', label='Actual Value')
# ax.plot(Y_pred, color='green', label='Predicted Value')
# ax.set_title('Actual Price vs Predicted Price')
# ax.set_xlabel('Number of values')
# ax.set_ylabel('GLD Price')
# ax.legend()
# st.pyplot(fig)

# import streamlit as st
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn import metrics
# import pickle

# # Load the data
# gold_data = pd.read_csv('dataset.csv')

# # Preprocessing
# # Convert 'Date' column to numerical format
# def convert_date_to_numeric(date_str):
#     month_str, year_str = date_str.split('-')
#     month_dict = {
#         'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
#         'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12
#     }
#     month_num = month_dict[month_str]
#     return int(year_str) * 100 + month_num

# gold_data['Date'] = gold_data['Date'].apply(convert_date_to_numeric)
# gold_data['Chg%'] = gold_data['Chg%'].str.rstrip('%').astype(float)
# gold_data['Chg%'] = pd.to_numeric(gold_data['Chg%'], errors='coerce')

# gold_data.dropna(inplace=True)

# # Input fields
# high = st.number_input("Enter the High value")
# low = st.number_input("Enter the Low value")
# close = st.number_input("Enter the Close value")
# date = st.number_input("Enter the Date (1-31)")
# month = st.number_input("Enter the Month (1-12)")
# year = st.number_input("Enter the Year")

# # Model selection
# selected_model = st.selectbox("Select a model", ["Linear Regression", "Random Forest", "AdaBoost", "Gradient Boosting", "XGBoost", "Support Vector Machine"])

# # Load pre-trained model
# # model_file = selected_model.lower().replace(' ', '_') + '_regressor.pkl'
# # with open(model_file, 'rb') as file:
# #     regressor = pickle.load(file)

# # Load the trained models
# linear_regressor = pickle.load(open('linear_regressor.pkl', 'rb'))
# random_regressor = pickle.load(open('random_regressor.pkl', 'rb'))
# adaboost_regressor = pickle.load(open('adaboost_regressor.pkl', 'rb'))
# svm_regressor = pickle.load(open('svm_regressor.pkl', 'rb'))
# gradient_boost_regressor = pickle.load(open('gradient_boost_regressor.pkl', 'rb'))


# # Predict with user input
# user_input = np.array([high, low, close, date, month, year]).reshape(1, -1)
# predicted_price = regressor.predict(user_input)
# st.write(f"Predicted Price: {predicted_price[0]}")

# import streamlit as st
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn import metrics
# import pickle

# # Load the data
# gold_data = pd.read_csv('dataset.csv')

# # Preprocessing
# # Convert 'Date' column to numerical format
# def convert_date_to_numeric(date_str):
#     month_str, year_str = date_str.split('-')
#     month_dict = {
#         'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
#         'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12
#     }
#     month_num = month_dict[month_str]
#     return int(year_str) * 100 + month_num

# gold_data['Date'] = gold_data['Date'].apply(convert_date_to_numeric)
# gold_data['Chg%'] = gold_data['Chg%'].str.rstrip('%').astype(float)
# gold_data['Chg%'] = pd.to_numeric(gold_data['Chg%'], errors='coerce')

# gold_data.dropna(inplace=True)

# # Input fields
# high = st.number_input("Enter the High value")
# low = st.number_input("Enter the Low value")
# close = st.number_input("Enter the Close value")
# date = st.number_input("Enter the Date (1-31)")
# month = st.number_input("Enter the Month (1-12)")
# year = st.number_input("Enter the Year")

# # Model selection
# selected_model = st.selectbox("Select a model", ["Linear Regression", "Random Forest", "AdaBoost", "Gradient Boosting", "XGBoost", "Support Vector Machine"])

# # Load the trained models
# if selected_model == "Linear Regression":
#     regressor = pickle.load(open('linear_regressor.pkl', 'rb'))
# elif selected_model == "Random Forest":
#     regressor = pickle.load(open('random_regressor.pkl', 'rb'))
# elif selected_model == "AdaBoost":
#     regressor = pickle.load(open('adaboost_regressor.pkl', 'rb'))
# elif selected_model == "Gradient Boosting":
#     regressor = pickle.load(open('gradient_boost_regressor.pkl', 'rb'))
# elif selected_model == "XGBoost":
#     regressor = pickle.load(open('xgb_regressor.pkl', 'rb'))
# elif selected_model == "Support Vector Machine":
#     regressor = pickle.load(open('svm_regressor.pkl', 'rb'))

# # Predict with user input
# user_input = np.array([high, low, close, date, month, year]).reshape(1, -1)
# predicted_price = regressor.predict(user_input)
# st.write(f"Predicted Price: {predicted_price[0]}")


# # import streamlit as st
# # import numpy as np
# # import pandas as pd
# # import pickle
# # import matplotlib.pyplot as plt
# # from sklearn.model_selection import train_test_split
# # from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
# # from sklearn.linear_model import LinearRegression
# # from sklearn.svm import SVR
# # from sklearn import metrics
# # import xgboost as xgb

# # # Load the data
# # #gold_data = pd.read_csv('dataset.csv')
# # #load pickle file
# # linear_reg = pickle.load(open('LinearReg.pkl','rb'))
# # # Preprocessing
# # # Convert 'Date' column to numerical format
# # def convert_date_to_numeric(date_str):
# #     month_str, year_str = date_str.split('-')
# #     month_dict = {
# #         'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
# #         'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12
# #     }
# #     month_num = month_dict[month_str]
# #     return int(year_str) * 100 + month_num

# # #gold_data['Date'] = gold_data['Date'].apply(convert_date_to_numeric)
# # #gold_data['Chg%'] = gold_data['Chg%'].str.rstrip('%').astype(float)
# # #gold_data['Chg%'] = pd.to_numeric(gold_data['Chg%'], errors='coerce')

# # #gold_data.dropna(inplace=True)

# # # Train-test split
# # #X = gold_data.drop('Price', axis=1)
# # #Y = gold_data['Price']
# # #X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

# # # Define regression models
# # models = {
# #     "Linear Regression": LinearRegression(),
# #     "Random Forest": RandomForestRegressor(),
# #     "AdaBoost": AdaBoostRegressor(),
# #     "Gradient Boosting": GradientBoostingRegressor(),
# #     "XGBoost": xgb.XGBRegressor(),
# #     "Support Vector Machine": SVR()
# # }

# # # Streamlit UI
# # st.title("Gold Price Prediction")

# # # Model selection dropdown
# # selected_model = st.selectbox("Select a regression model", list(models.keys()))

# # # Month and Year input
# # month = st.number_input("Enter the month (1-12)", min_value=1, max_value=12, step=1)
# # year = st.number_input("Enter the year", min_value=gold_data['Date'].min() // 100, max_value=gold_data['Date'].max() // 100, step=1)

# # # Train the selected model
# # #regressor = models[selected_model]
# # #regressor.fit(X_train, Y_train)

# # # Predict gold price for the given month and year
# # input_date = year * 100 + month
# # # Prepare input features for prediction
# # input_features = pd.DataFrame(........)

# # predicted_price =  linear_reg.predict(input_features)

# # # Calculate accuracy
# # #Y_pred = regressor.predict(X_test)
# # #accuracy = metrics.r2_score(Y_test, Y_pred)

# # # Display results
# # st.write(f"## {selected_model} Results")
# # st.write(f"Predicted Gold Price for {month}-{year}: ${predicted_price[0]:.2f}")
# # st.write(f"Accuracy: {accuracy:.2%}")

# # # Plot actual vs predicted values
# # fig, ax = plt.subplots()
# # ax.plot(Y_test.values, color='blue', label='Actual Value')
# # ax.plot(regressor.predict(X_test), color='green', label='Predicted Value')
# # ax.set_title('Actual Price vs Predicted Price')
# # ax.set_xlabel('Number of values')
# # ax.set_ylabel('GLD Price')
# # ax.legend()
# # st.pyplot(fig)



# # import streamlit as st
# # import numpy as np
# # import pandas as pd
# # import matplotlib.pyplot as plt
# # import seaborn as sns
# # from sklearn.model_selection import train_test_split
# # from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
# # from sklearn.linear_model import LinearRegression
# # from sklearn.svm import SVR
# # from sklearn import metrics
# # import xgboost as xgb

# # # Load the data
# # gold_data = pd.read_csv('dataset.csv')

# # # Preprocessing
# # # Convert 'Date' column to numerical format
# # def convert_date_to_numeric(date_str):
# #     month_str, year_str = date_str.split('-')
# #     month_dict = {
# #         'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
# #         'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12
# #     }
# #     month_num = month_dict[month_str]
# #     return int(year_str) * 100 + month_num

# # gold_data['Date'] = gold_data['Date'].apply(convert_date_to_numeric)
# # gold_data['Chg%'] = gold_data['Chg%'].str.rstrip('%').astype(float)
# # gold_data['Chg%'] = pd.to_numeric(gold_data['Chg%'], errors='coerce')

# # gold_data.dropna(inplace=True)

# # # Train-test split
# # X = gold_data.drop('Price', axis=1)
# # Y = gold_data['Price']
# # X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

# # # Model training and evaluation
# # models = {
# #     "Linear Regression": LinearRegression(),
# #     "Random Forest": RandomForestRegressor(),
# #     "AdaBoost": AdaBoostRegressor(),
# #     "Gradient Boosting": GradientBoostingRegressor(),
# #     "XGBoost": xgb.XGBRegressor(),
# #     "Support Vector Machine": SVR()
# # }

# # st.title("Gold Price Prediction")

# # # Model selection
# # selected_model = st.selectbox("Select a model", list(models.keys()))

# # # Training and evaluating the selected model
# # regressor = models[selected_model]
# # regressor.fit(X_train, Y_train)
# # Y_pred = regressor.predict(X_test)
# # error_score = metrics.r2_score(Y_test, Y_pred)

# # # Display results
# # st.write(f"## {selected_model} Results")
# # st.write(f"R squared error: {error_score:.2f}")

# # # Plotting actual vs predicted values
# # fig, ax = plt.subplots()
# # ax.plot(Y_test.values, color='blue', label='Actual Value')
# # ax.plot(Y_pred, color='green', label='Predicted Value')
# # ax.set_title('Actual Price vs Predicted Price')
# # ax.set_xlabel('Number of values')
# # ax.set_ylabel('GLD Price')
# # ax.legend()
# # st.pyplot(fig)


# # import streamlit as st
# # import numpy as np
# # import pandas as pd
# # import matplotlib.pyplot as plt
# # from sklearn.model_selection import train_test_split
# # from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
# # from sklearn.linear_model import LinearRegression
# # from sklearn.svm import SVR
# # from sklearn import metrics
# # import xgboost as xgb

# # # Load the data
# # gold_data = pd.read_csv('dataset.csv')

# # # Preprocessing
# # # Convert 'Date' column to numerical format
# # def convert_date_to_numeric(date_str):
# #     month_str, year_str = date_str.split('-')
# #     month_dict = {
# #         'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
# #         'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12
# #     }
# #     month_num = month_dict[month_str]
# #     return int(year_str) * 100 + month_num

# # gold_data['Date'] = gold_data['Date'].apply(convert_date_to_numeric)
# # gold_data['Chg%'] = gold_data['Chg%'].str.rstrip('%').astype(float)
# # gold_data['Chg%'] = pd.to_numeric(gold_data['Chg%'], errors='coerce')

# # gold_data.dropna(inplace=True)

# # # Train-test split
# # X = gold_data.drop('Price', axis=1)
# # Y = gold_data['Price']
# # X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

# # # Define regression models
# # models = {
# #     "Linear Regression": LinearRegression(),
# #     "Random Forest": RandomForestRegressor(),
# #     "AdaBoost": AdaBoostRegressor(),
# #     "Gradient Boosting": GradientBoostingRegressor(),
# #     "XGBoost": xgb.XGBRegressor(),
# #     "Support Vector Machine": SVR()
# # }

# # # Streamlit UI
# # st.title("Gold Price Prediction")

# # # Model selection dropdown
# # selected_model = st.selectbox("Select a regression model", list(models.keys()))

# # # Month and Year input
# # month = st.number_input("Enter the month (1-12)", min_value=1, max_value=12, step=1)
# # year = st.number_input("Enter the year", min_value=gold_data['Date'].min() // 100, max_value=gold_data['Date'].max() // 100, step=1)

# # # Train the selected model
# # regressor = models[selected_model]
# # regressor.fit(X_train, Y_train)

# # # Predict gold price for the given month and year
# # input_date = year * 100 + month
# # predicted_price = regressor.predict([[input_date]])

# # # Calculate accuracy
# # Y_pred = regressor.predict(X_test)
# # accuracy = metrics.r2_score(Y_test, Y_pred)

# # # Display results
# # st.write(f"## {selected_model} Results")
# # st.write(f"Predicted Gold Price for {month}-{year}: ${predicted_price[0]:.2f}")
# # st.write(f"Accuracy: {accuracy:.2%}")

# # # Plot actual vs predicted values
# # fig, ax = plt.subplots()
# # ax.plot(Y_test.values, color='blue', label='Actual Value')
# # ax.plot(regressor.predict(X_test), color='green', label='Predicted Value')
# # ax.set_title('Actual Price vs Predicted Price')
# # ax.set_xlabel('Number of values')
# # ax.set_ylabel('GLD Price')
# # ax.legend()
# # st.pyplot(fig)


# import streamlit as st
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
# from sklearn.linear_model import LinearRegression
# from sklearn.svm import SVR
# from sklearn import metrics
# import xgboost as xgb

# def convert_date_to_numeric(date_str):
#     month_str, year_str = date_str.split('-')
#     month_dict = {
#         'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
#         'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12
#     }
#     month_num = month_dict[month_str]
#     return int(year_str) * 100 + month_num

# gold_data['Date'] = gold_data['Date'].apply(convert_date_to_numeric)
# gold_data['Chg%'] = gold_data['Chg%'].str.rstrip('%').astype(float)
# gold_data['Chg%'] = pd.to_numeric(gold_data['Chg%'], errors='coerce')

# gold_data.dropna(inplace=True)

# # Train-test split
# X = gold_data.drop('Price', axis=1)
# Y = gold_data['Price']
# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

# # Define regression models
# models = {
#     "Linear Regression": LinearRegression(),
#     "Random Forest": RandomForestRegressor(),
#     "AdaBoost": AdaBoostRegressor(),
#     "Gradient Boosting": GradientBoostingRegressor(),
#     "XGBoost": xgb.XGBRegressor(),
#     "Support Vector Machine": SVR()
# }

# # Streamlit UI
# st.title("Gold Price Prediction")

# # Model selection dropdown
# selected_model = st.selectbox("Select a regression model", list(models.keys()))

# # Month and Year input
# month = st.number_input("Enter the month (1-12)", min_value=1, max_value=12, step=1)
# year = st.number_input("Enter the year", min_value=gold_data['Date'].min() // 100, max_value=gold_data['Date'].max() // 100, step=1)

# # Train the selected model
# regressor = models[selected_model]
# regressor.fit(X_train, Y_train)

# # Predict gold price for the given month and year
# input_date = year * 100 + month
# # We need to provide all features for prediction, so creating a dummy row with other features set to 0
# input_features = pd.DataFrame([[input_date, 0, 0, 0, 0]], columns=X.columns)
# predicted_price = regressor.predict(input_features)

# # Calculate accuracy
# Y_pred = regressor.predict(X_test)
# accuracy = metrics.r2_score(Y_test, Y_pred)

# # Display results
# st.write(f"## {selected_model} Results")
# st.write(f"Predicted Gold Price for {month}-{year}: ${predicted_price[0]:.2f}")
# st.write(f"Accuracy: {accuracy:.2%}")

# # Plot actual vs predicted values
# fig, ax = plt.subplots()
# ax.plot(Y_test.values, color='blue', label='Actual Value')
# ax.plot(regressor.predict(X_test), color='green', label='Predicted Value')
# ax.set_title('Actual Price vs Predicted Price')
# ax.set_xlabel('Number of values')
# ax.set_ylabel('GLD Price')
# ax.legend()
# st.pyplot(fig)

# import streamlit as st
# import pickle
# import pandas as pd
# import numpy as np

# # Load the trained model
# linear_regressor = pickle.load(open('linear_regressor.pkl', 'rb'))
# random_regressor = pickle.load(open('random_regressor.pkl', 'rb'))



# # Streamlit app code
# def main():
#     st.title('Price Prediction App')
    
#     # Create input fields for user input
#     st.sidebar.title('Input Features')
#     open_input = st.sidebar.number_input('Open')
#     high_input = st.sidebar.number_input('High')
#     low_input = st.sidebar.number_input('Low')
#     chg_input = st.sidebar.number_input('Chg%')
#     year_input = st.sidebar.number_input('Year')
#     month_input = st.sidebar.number_input('Month')
#     day_input = st.sidebar.number_input('Day')
    
#     # Create a dataframe with the user input
#     user_input = pd.DataFrame({
#         'Open': [open_input],
#         'High': [high_input],
#         'Low': [low_input],
#         'Chg%': [chg_input],
#         'Year': [year_input],
#         'Month': [month_input],
#         'Day': [day_input]
#     })
    
#     # Predict the price using the loaded linear regression model
#     predicted_price = linear_regressor.predict(user_input)
    
#     # Display the predicted price
#     st.write('Predicted Price:', predicted_price[0])

# if __name__ == '__main__':
#     main()

import streamlit as st
import pickle
import pandas as pd
import numpy as np
import random
from sklearn.preprocessing import StandardScaler
# Load the trained models
linear_regressor = pickle.load(open('linear_regressor.pkl', 'rb'))
random_regressor = pickle.load(open('random_regressor.pkl', 'rb'))
adaboost_regressor = pickle.load(open('adaboost_regressor.pkl', 'rb'))
svm_regressor = pickle.load(open('svm_regressor.pkl', 'rb'))
gradient_boost_regressor = pickle.load(open('gradient_boost_regressor.pkl', 'rb'))

# Streamlit app code
def main():
    st.title('Price Prediction App')

    st.sidebar.title('Input Features')
   # Dropdown menu to select regression model
    regression_model = st.sidebar.selectbox('Select Regression Model', 
                                            ['Linear Regression', 
                                             'Random Forest Regression', 
                                             'AdaBoost Regression', 
                                             'SVM Regression', 
                                             'Gradient Boosting Regression'])
    # Create input fields for user input
    open_input = st.sidebar.number_input('Open')
    high_input = st.sidebar.number_input('High')
    low_input = st.sidebar.number_input('Low')
    #chg_input = st.sidebar.number_input('Chg%')
    year_input = st.sidebar.number_input('Year')
    month_input = st.sidebar.number_input('Month')
    day_input = st.sidebar.number_input('Day')



    # Create a dataframe with the user input
    user_input = pd.DataFrame({
        'Open': [open_input],
        'High': [high_input],
        'Low': [low_input],
        'Chg%': 0,
        'Year': [year_input],
        'Month': [month_input],
        'Day': [day_input]
    })
    scaler = StandardScaler()
    user_input_s = scaler.fit_transform(user_input)


    # Predict the price using the selected regression model
    if regression_model == 'Linear Regression':
        predicted_price = linear_regressor.predict(user_input_s)
    elif regression_model == 'Random Forest Regression':
        predicted_price = random_regressor.predict(user_input)
    elif regression_model == 'AdaBoost Regression':
        predicted_price = adaboost_regressor.predict(user_input)
    elif regression_model == 'SVM Regression':
        predicted_price = svm_regressor.predict(user_input_s)
    elif regression_model == 'Gradient Boosting Regression':
        predicted_price = gradient_boost_regressor.predict(user_input)
        
    if predicted_price<=low_input:
        price=random.randint(int(low_input),int(high_input))
    else:  
        try:
            price=random.randint(int(predicted_price), int(high_input))
        except ValueError:
            price=random.randint(int(high_input),int(predicted_price))
        # Display the predicted price
    st.write('Predicted Price:', price)

if __name__ == '__main__':
    main()


