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

pitching_data.set_index(["name", "season"], inplace = True) # set the index
sorted_pitching_data = pitching_data.groupby('name', group_keys=False).apply(lambda x: x.sort_values('season')) # what does this do??? sorts each group by season
sorted_pitching_data['last_year_babip'] = sorted_pitching_data.groupby('name')['babip'].shift(1) # explanation of line:

'''
We group by 'name' only in the second step because we want to shift the 'babip' values within each player's data.
If we grouped by both 'name' and 'season', each group would only have one row 
(since each player has only one row per season), so there would be nothing to shift.

The shift operation in pandas works such that a positive value shifts data downwards 
(i.e., towards later rows), while a negative value shifts data upwards (i.e., towards earlier rows). 
So, shift(-1) moves each 'babip' value one row up, effectively making it appear on the previous season's row. 
This is how we get the next season's 'babip' on the current season's row.

Changing the value to shift(1) would take the babip from the previous year
'''

contact_quality_columns = ["babip", "last_year_babip", "gb/fb", "ld%", "gb%", "fb%", "iffb%", "hr/fb", "pull%", "cent%", "oppo%", "soft%", "med%", "hard%"] # select contact quality columns
contact_data = sorted_pitching_data[contact_quality_columns] # select only the contact stat columns
contact_data = contact_data.dropna()

# split the data into feature and target data
X = contact_data[['last_year_babip']] # only last year BABIP
#print(X)
y = contact_data.babip # to determine this year BABIP
#print(y)

# split the data into testing and training data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42) # split the data

# create a model
model = LinearRegression()

# train the model
model.fit(X_train, y_train)

# use the model to make a prediciton using the testing data
y_pred = model.predict(X_test) # get the prediction data

# plot the data
plt.scatter(y_test, y_pred) # plot the test (actual) data and the predicted data from the model
plt.xlabel('Actual BABIP')
plt.ylabel('Predicted BABIP')
plt.title('Actual vs Predicted BABIP (B2B)')
r2 = r2_score(y_test, y_pred)
print(f"B2B V2, R^2 = {r2}")
plt.text(.3, .26, f"B2B V2, r^2 = {r2}") # show the r^2
plt.text(.3, .265, f"Num Samples: {X.shape[0]}")
print(f"Num Samples: {X.shape[0]}")
#plt.show()

