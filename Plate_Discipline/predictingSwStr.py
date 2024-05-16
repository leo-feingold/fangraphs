import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

swing_decision_data = pd.read_csv("/Users/leofeingold/Desktop/fangraphs/Plate_Discipline/fangraphs-leaderboards (2).csv").rename(columns = lambda x: x.lower()) #load the data and make the column names lower case
swing_decision_data.set_index(["name", "season"], inplace=True) # index by name and season
sorted_swing_decision = swing_decision_data.groupby("name", group_keys=False).apply(lambda x: x.sort_values("season", ascending=True)) # group by name and sort each group by season 
sorted_swing_decision["next_year_swstr"] = sorted_swing_decision.groupby("name")["swstr%"].shift(-1) # add the next season's swstr% to the current season's row -> this is what we will want to predict
swing_decision_columns = ["o-swing%", "z-swing%", "swing%", "o-contact%", "z-contact%", "contact%", "zone%", "f-strike%", "swstr%", "next_year_swstr", "cstr%", "csw%", "bb%", "k%"] # select the columns we want
df = sorted_swing_decision[swing_decision_columns] # apply the column selection
copy_df = df # make copy that still contains na values for later on
df = df.dropna() # drop na values


#perform linear regression to predict next season swstr%:
#split the data into feature and target
X = df.drop("next_year_swstr", axis = 1)
y = df.next_year_swstr

#split data into training and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

#create the predictive model
model = LinearRegression()

#train the model
model.fit(X_train, y_train)

#create predicitions with the now-trained model
y_pred = model.predict(X_test)

#calculate the r^2
r2 = r2_score(y_test, y_pred)
numSamples = X.shape[0]
#print("Description: Actual VS Predicted SwStr% in Hitters")
#print(f"R^2 = {r2}")
#print(f"Num Samples: {numSamples}")

#plot the data
plt.scatter(y_test, y_pred)
plt.xlabel('Actual SwStr%')
plt.ylabel('Predicted SwStr%')
plt.title('Actual VS Predicted SwStr% in Hitters')
plt.suptitle(f"r^2 = {r2}, Num Samples: {numSamples}")
#plt.show()

#get the coeffeicents and equation from the model
equation_terms = [f"{coef}*{feature}" for feature, coef in zip(X.columns, model.coef_)]
equation = " + ".join(equation_terms) + f" + {model.intercept_}"
#print("Equation From Model:")
#print(equation)

'''
Output: (Note: each percentage is stored as a decimal)

0.03785649353770773*o-swing% + 0.012048926995733317*z-swing% +
 -0.049405763425942786*swing% + -0.0037284839217076524*o-contact% + 
 0.06310776616415868*z-contact% + -0.11972649482757614*contact% + 
 -0.022409576843383447*zone% + 0.022299579244944672*f-strike% + 
 10.98355980875476*swstr% + 10.281332812444882*cstr% + -10.344480727878416*csw% + 
 -0.010616415633045079*bb% + 0.028343274896604934*k% + 0.08401622655271755
'''

def sample_player(sample_player, sample_year):
    player_data = copy_df.loc[(sample_player, sample_year)] # we can use this syntax because the name and season are indeces of the df
    return player_data.to_dict()


def test_model(oswing, zswing, swing, ocontact, zcontact, contact, zone, fstrike, swstr, cstr, csw, bb, k):
    return 0.03785649353770773*(oswing) + 0.012048926995733317*(zswing) + -0.049405763425942786*(swing) + -0.0037284839217076524*(ocontact) + 0.06310776616415868*(zcontact) + -0.11972649482757614*(contact) + -0.022409576843383447*(zone) + 0.022299579244944672*(fstrike) + 10.98355980875476*(swstr) + 10.281332812444882*(cstr) + -10.344480727878416*(csw) +  -0.010616415633045079*(bb) + 0.028343274896604934*(k) + 0.08401622655271755

desired_player = "Mookie Betts"
desired_year_to_project = 2023 #dtype must be int
data = sample_player(desired_player, desired_year_to_project - 1)

# Predicting swstr%, the order here is very important
predicted_swstr = test_model(
    data["o-swing%"], data["z-swing%"], data["swing%"], 
    data["o-contact%"], data["z-contact%"], data["contact%"], 
    data["zone%"], data["f-strike%"], data["swstr%"], 
    data["cstr%"], data["csw%"], data["bb%"], data["k%"]
)

print(f"Predicted SwStr% for {desired_player} in the year {desired_year_to_project}: {predicted_swstr}")