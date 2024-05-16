import mplcursors 
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

player_test_val = "Yu Darvish"
testChar = "stf+ fa"
valueChar = "wfa (sc)"
pitchData = pd.read_csv("/Users/leofeingold/Desktop/fangraphs/pitch_design/new.csv").rename(columns = lambda x: x.lower())
pitchData.dropna()

#pitchData = pd.read_csv("/Users/leofeingold/Desktop/fangraphs/pitch_design/fangraphs-leaderboards (4).csv").rename(columns = lambda x: x.lower())
fastballData = pitchData[["name", "team", "ip", "season", "age", "fa% (sc)", "vfa (sc)", "fa-x (sc)", "fa-z (sc)", "wfa/c (sc)", "stf+ fa", "mlbamid", "wfa (sc)"]]
cole_data1 = fastballData.query(f"name == '{player_test_val}'")
print(f"{player_test_val}:")
print(1)
print(cole_data1[['season', 'stf+ fa']])


savantData = pd.read_csv("/Users/leofeingold/Desktop/fangraphs/pitch_design/stats.csv").rename(columns = lambda x: x.lower())
savantData.dropna()
savantFastball = savantData[['last_name, first_name', 'player_id', 'year', 'woba', 'xwoba', 'ff_avg_speed', 'ff_avg_spin', 'ff_avg_break_x', 'ff_avg_break_z', 'ff_avg_break']]


merged_df = pd.merge(fastballData, savantFastball, left_on=['mlbamid', 'season'], right_on=['player_id', 'year'], how='inner')
cole_data2 = merged_df.query(f"name == '{player_test_val}'")
print(2)
print(cole_data2[['season', 'stf+ fa']])

minUsage = 0.0
minVelo = 92.5
maxVelo = 94.5
mask = (merged_df['fa% (sc)'] > minUsage) & (merged_df['vfa (sc)'] > minVelo) & (merged_df['vfa (sc)'] < maxVelo)
merged_df = merged_df[mask]


cole_data3 = merged_df.query(f"name == '{player_test_val}'")
print(3)
print(cole_data3[['season', 'stf+ fa']])

corr = merged_df[testChar].corr(merged_df[valueChar])
plt.scatter(merged_df[testChar], merged_df[valueChar])

cursor = mplcursors.cursor(hover=True)
cursor.connect("add", lambda sel: sel.annotation.set_text(
   f"Pitcher: {merged_df['name'].iloc[sel.index]}\nSeason: {merged_df['season'].iloc[sel.index]}"
))

plt.xlabel(f"{testChar}")
plt.ylabel(f"{valueChar}")
plt.suptitle(f"Correlation: {corr}, Sample: {len(merged_df)}")
plt.title(f"{testChar} vs {valueChar} (2021-2023), Minimum Usage {minUsage}, Minimum Velo: {minVelo}, Maximum Velo: {maxVelo}")
plt.show()