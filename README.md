hello folks :) 

This is my minor project for the sixth semester. In this project with the help of machine learning algorithms primarily regression techniques like linear regression, Random Forest regression, AdaBoost Regression, XGboost regression, Support vector regression and Gradient boost regression we have tried to predict future gold prices.

## DATASET

Dataset we used in the process is saved by "__golddaily.csv__". It has daily gold price from 01 january 2014 to 01 january 2024. It has attributes of Price, Open, High, Low and change% . Source of dataset is INVESTING.COM and WORLD GOLD COUNCIL.

## Libraries used

numpy 
pandas
seaborn
matplotlib
sklearn(model_selection, ensemble, metrics)
pickle
streamlit

## Data preprocessing

data was checked for any kind of null or garbage values using `isnull()`function. Dataset present was not of enough size to train model efficiently so we created a synthetic dataframe of 100 inputs based out of original dataset. Both original and synthetic datasets were concatenated in dataframe 'merged_data'.

## Data Extraction

To make it correlatable we converted all non int or float datatype attributes to int64 and change% to float64. we would calculate correlation among attributes. this correlation would be further helpful in model training.
we would then assign values X=merged_data.drop(all attribute execpt Price) and Y = merged_data(price).
then we would split merged_data dataframe in training set and test set using train_test_split from sklearn's model_selection. We would then Standardise the data using StandardScaler to avoid outliers from effecting our predictions.

## model Training

we would first start by running linear regression model by using LinearRegression() function. then print the prediction. then we would print the metrics of predictions. we would then print the visualisation charts of the results.

we would repeat the same for Random forest, Xgboost, Adaboost, and gradient boost.

for SVM we would tune data on basis of hyperparameter (hyperparameter tuning). This helps in selection best feature and handle data imbalance. then we would print the metrics result and visualise data.

##result and conclusion

as all models have been trained we would print combined metrics of all together and compare the accuracy of various models. we would then visualise all the models for a better explanations of insight

##Deployment

we would deploy the model using streamlit. As you can see we had earlier made pickle file for each model where their individual predictions would be save. we ould import all the pickle files. we would then create a frame work for our show website. Then we have to connect frontend characters and input variables with backend attributes in order to make website fully functional. we would be giving dropdown menu for allowing user to choose his/her model for prediction. 

final step is to run our streamlit.py file using command line.

/////////////////////////////////////////////////////// the project is now completed ///////////////////////////////////////////////////
