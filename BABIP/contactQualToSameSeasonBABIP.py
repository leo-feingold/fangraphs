'''
My goal (for now) is to create a predictive model that, given the contact quality statistics, can predict the expected babip.
Then I want to find over and underperformers.
'''

# import libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import scipy.optimize
from sklearn.metrics import r2_score

# load and process data
pitching_data = pd.read_csv("/Users/leofeingold/Desktop/fangraphs/BABIP/fangraphs-leaderboards.csv").rename(columns = lambda x: x.lower()) # load the data with lowercase column names
pitching_data_sorted = pitching_data.sort_values('babip', ascending = False) # sort by descending babip
contact_quality_columns = ["name", "season", "babip", "gb/fb", "ld%", "gb%", "fb%", "iffb%", "hr/fb", "pull%", "cent%", "oppo%", "soft%", "med%", "hard%"] # select contact quality columns
contact_data = pitching_data_sorted[contact_quality_columns] # select only contact columns
contact_data.set_index(["name", "season"], inplace = True) # take the non statistical columns (name and for now, season) and make them indeces instead of columns
num_features = len(contact_data.columns)
#print(num_features)

# split data into feature and target data
X = contact_data.drop('babip', axis = 1) # features are all the columns except for babip, axis = 1 implies dropping a column
y = contact_data.babip # target is babip

# randomely shuffle the data and break it up into trianing and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42) # 20% testing data

# create a model
model = LinearRegression() # use built in function

# train the model
model.fit(X_train, y_train) # train the model by fitting it to the training data

# use the model to make a prediciton using the testing data
y_pred = model.predict(X_test) # get the prediction data

# create a scatter plot
plt.scatter(y_test, y_pred) # plot the test (actual) data and the predicted data from the model
plt.xlabel('Actual BABIP Values')
plt.ylabel('Predicted BABIP Values')
plt.title('Actual vs Predicted BABIP (kinda ass)')
#r2 = model.score(X_test, y_test) #calculate the r^2
r2 = r2_score(y_test, y_pred)
print(f"Multi Lin Reg v1, R^2 = {r2}")
print(f"Num Samples: {X.shape[0]}")
plt.text(.3, .25, f"r^2: {r2}") # show the r^2
plt.text(.3, .254, f"LinReg v1, Num Samples: {X.shape[0]}")
plt.show()

# look at individual inputs coorelation with babip
correlations = contact_data.corr().babip 
#print(correlations.sort_values(ascending = True))

