# import libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import scipy.optimize
from sklearn.metrics import r2_score
import mplcursors 

# load and process data
pitching_data = pd.read_csv("/Users/leofeingold/Desktop/fangraphs/BABIP/fangraphs-leaderboards.csv").rename(columns = lambda x: x.lower()) # load the data with lowercase column names
pitching_data.set_index(["name", "season"], inplace = True)
sorted_pitching_data = pitching_data.groupby('name', group_keys=False).apply(lambda x: x.sort_values('season', ascending = True)) # what does this do??? sorts each group by season
sorted_pitching_data['next_year_babip'] = sorted_pitching_data.groupby('name')['babip'].shift(-1) # add the next year babip to the row, this is what we want to predict

#print(sorted_pitching_data.loc["Zack Wheeler"][["babip", "next_year_babip"]]) #Note: this looks good, we want to use contact quality and babip (in say 2018) to predict babip (in 2019, or next qualified season)

contact_quality_columns = ["babip", "next_year_babip", "gb/fb", "ld%", "gb%", "fb%", "iffb%", "hr/fb", "pull%", "cent%", "oppo%", "soft%", "med%", "hard%"] # select contact quality columns
contact_data = sorted_pitching_data[contact_quality_columns] # select only the contact stat columns
contact_data = contact_data.dropna()

#print(contact_data.loc["Zack Wheeler"]) #Note: this also looks correct, data is filtered to have neccesary columns, there are no NaN values


# plot some data
player_season_index = contact_data.index
players = [index[0] for index in player_season_index]  # Player names
seasons = [index[1] for index in player_season_index]  # Seasons

# Create scatter plot
plt.scatter(contact_data["fb%"], contact_data["babip"])

# Add cursor tooltips
cursor = mplcursors.cursor(hover=True)
cursor.connect("add", lambda sel: sel.annotation.set_text(
    f"{players[sel.index]}\nSeason: {seasons[sel.index]}"
))

plt.xlabel('fb')
plt.ylabel('BABIP')
plt.title('FB% vs BABIP')
plt.show()


# Now do the regression:
# split the data into feature and target data
X = contact_data.drop("next_year_babip", axis = 1) #feature data, everything except for the next year's babip because that's what we are trying to predict
y = contact_data.next_year_babip #target data, the next year's babip

# split the data into training and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42)

# create the linear regression model
model = LinearRegression()

# train the model
model.fit(X_train, y_train)

# make predictions with the now trained model
y_pred = model.predict(X_test)

# calculate the r^2
r2 = r2_score(y_test, y_pred)
print("Description: With contact quality and BABIP for a given year, predict the BABIP next year:")
print(f"R^2 = {r2}")
print(f"Num Samples: {X.shape[0]}")

#Get Formula
#for feature, coef in zip(X.columns, model.coef_):
    #print(f"{feature}*{coef} + ")
#print(model.intercept_)

'''
Expected BABIP (xBABIP) Formula, r^2 ≈ 0.06176 (pretty bad):

babip*0.0015129527926563185 + gb/fb*-0.012640783922911038 + ld%*-2808713.4428602983 + gb%*-2808713.3385114 + 
fb%*-2808713.5349376667 + iffb%*-0.00648309484346612 + hr/fb*0.016562431026914647 + pull%*20.986364022279602 + 
cent%*20.925979305006404 + oppo%*21.069216092802108 + soft%*21.947713521233652 + med%*22.067640843240515 + 
hard%*22.04359333751631 + 2808670.7013290105
'''