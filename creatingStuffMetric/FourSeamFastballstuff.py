import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mplcursors
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler 

fastballData = pd.read_csv("/Users/leofeingold/Desktop/fangraphs/creatingStuffMetric/Baseball Savant StatCast Search Stuff+ (Min 200 PA) - 4-Seam Fastball  Data (2023-2019, excluding 2020) (1).csv").rename(columns = lambda x: x.lower())
fastballData['xwoba/average xwoba of year'] = fastballData['xwoba/average xwoba of year'] * 100

'''
fastballData = fastballData[['player', 'year', 
       'glove/arm-side movement (in)', 'vertical movement w/o gravity (in)',
       'woba', 'xwoba', 'babip', 'whiff%', 'pitch (mph)', 'perceived velocity',
       'spin (rpm)', 'vertical release pt (ft)', 'horizontal release pt (ft)',
       'extension (ft)', 'xwoba/average xwoba of year']]
'''

fastballData = fastballData[['player', 'year', 
       'glove/arm-side movement (in)', 'vertical movement w/o gravity (in)',
       'woba', 'xwoba', 'babip', 'whiff%', 'pitch (mph)', 'perceived velocity',
       'spin (rpm)', 'vertical release pt (ft)', 'horizontal release pt (ft)',
       'extension (ft)', 'xwoba/average xwoba of year']]

#print(fastballData['spin (rpm)'].corr(fastballData['xwoba/average xwoba of year']))

fastballData.set_index(["player", "year"], inplace = True)


# Convert the "glove/arm-side movement (in)" column to string type
fastballData['glove/arm-side movement (in)'] = fastballData['glove/arm-side movement (in)'].astype(str)
# Split the string on whitespace and extract the first part
movement_values = fastballData['glove/arm-side movement (in)'].str.split().str[0]
# Convert the extracted values to float
movement_values = movement_values.astype(float)
# Determine the sign adjustment based on whether "GLV" or "ARM" is present
sign_adjustment = np.where(fastballData['glove/arm-side movement (in)'].str.contains('GLV'), -1, 1)
# Apply the sign adjustment to the movement values
movement_values *= sign_adjustment
# Update the column with the adjusted movement values
fastballData['glove/arm-side movement (in)'] = movement_values
# visualize the data
#print(fastballData['glove/arm-side movement (in)'].sort_values(ascending = True))


X = fastballData.drop(["xwoba", "woba", "babip", "xwoba/average xwoba of year"], axis = 1)
y = fastballData["xwoba/average xwoba of year"]
numSamples = X.shape[0]

testSize = 0.1
randomState = 42
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = testSize, random_state = randomState)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LinearRegression()
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)
r2 = r2_score(y_test, y_pred)

# Plotting the y=x line here
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--', label='y=x')


plt.scatter(y_test, y_pred)
plt.xlabel('Actual xWOBA-')
plt.ylabel('Predicted xwOBA-')
plt.title('Actual VS Predicted xWOBA- on Fastball Pitch Characteristics')
plt.suptitle(f"r^2 = {r2}, Num Samples: {numSamples}, Test Size: {testSize}, Random State: {randomState}")
plt.gca().set_aspect('equal', adjustable='box')

cursor = mplcursors.cursor(hover=True)
cursor.connect("add", lambda sel: sel.annotation.set_text(
    f"Name: {fastballData.index[sel.index][0]}\nSeason: {fastballData.index[sel.index][1]}"
))

plt.show()


equation_terms = [f"{coef}*{feature}" for feature, coef in zip(X.columns, model.coef_)]
equation = " + ".join(equation_terms) + f" + {model.intercept_}"
print(equation)


player_name = "Lynn, Lance RHP"
year_to_predict = 2023  

player_data = fastballData.loc[(player_name, year_to_predict), X.columns]
player_data_scaled = scaler.transform(player_data.values.reshape(1, -1))

predicted_xwoba_minus = model.predict(player_data_scaled)
print(f"Predicted xwOBA- for {player_name}'s fastball in {year_to_predict}: {predicted_xwoba_minus[0]}")
print(f"Actual xwOBA- for {player_name}'s fastball in {year_to_predict}: {fastballData.loc[(player_name, year_to_predict)]['xwoba/average xwoba of year']}")