import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mplcursors

steamer = pd.read_csv("/Users/leofeingold/Desktop/fangraphs/projectionSystem/steamer2024.csv").rename(columns = lambda x: x + "_steamer")
steamer['steamer_index'] = steamer.index
#print(steamer.columns)

zips = pd.read_csv("/Users/leofeingold/Desktop/fangraphs/projectionSystem/zips2024.csv").rename(columns = lambda x: x + "_zips")
zips['zips_index'] = zips.index
#print(zips.columns)

atc = pd.read_csv("/Users/leofeingold/Desktop/fangraphs/projectionSystem/atc2024.csv").rename(columns = lambda x: x + "_atc")
#print(atc.columns)

thebat = pd.read_csv("/Users/leofeingold/Desktop/fangraphs/projectionSystem/thebat2024.csv").rename(columns = lambda x: x + "_thebat")
#print(thebat.columns)

zipsdc = pd.read_csv("/Users/leofeingold/Desktop/fangraphs/projectionSystem/zipsdc2024.csv").rename(columns = lambda x: x + "_zipsdc")
#print(zipsdc.columns)

depthcharts = pd.read_csv("/Users/leofeingold/Desktop/fangraphs/projectionSystem/depthcharts.csv").rename(columns = lambda x: x + "_depthcharts")
#print(depthcharts.columns)

# merge all the different projection system into one dataframe
zipsSteamer = pd.merge(zips, steamer, left_on = 'PlayerId_zips', right_on = 'PlayerId_steamer')
combo2 = pd.merge(atc, thebat, left_on = 'PlayerId_atc', right_on = 'PlayerId_thebat')
combo3 = pd.merge(zipsdc, depthcharts, left_on = 'PlayerId_zipsdc', right_on = 'PlayerId_depthcharts')
combo4 = pd.merge(zipsSteamer, combo2, left_on = 'PlayerId_zips', right_on = 'PlayerId_atc')
projections = pd.merge(combo3, combo4, left_on = 'PlayerId_zipsdc', right_on = 'PlayerId_zips')
projections.to_csv('projectionsBetter.csv', index = True)

zipsSteamer = zipsSteamer.assign(
    FIPdifference = lambda x: abs(x.FIP_zips - x.FIP_steamer)
)

zipsSteamer = zipsSteamer.sort_values('FIPdifference', ascending = False)

zipsSteamer = zipsSteamer[['zips_index', 'steamer_index', 'Name_zips', 'Team_zips','ERA_zips', 'G_zips', 'IP_zips', 'ER_zips', 'HR_zips', 'BB_zips', 'SO_zips', 'K/9_zips', 'BB/9_zips', 'K/BB_zips',
       'HR/9_zips', 'K%_zips', 'BB%_zips', 'K-BB%_zips', 'AVG_zips',
       'WHIP_zips', 'BABIP_zips', 'LOB%_zips', 'GB%_zips', 'HR/FB_zips',
       'FIP_zips', 'WAR_zips', 'RA9-WAR_zips', 'PlayerId_zips', 'Name_steamer',
       'Team_steamer', 'ERA_steamer',
       'G_steamer', 'IP_steamer', 'ER_steamer', 'HR_steamer',
       'BB_steamer', 'SO_steamer', 'K/9_steamer',
       'BB/9_steamer', 'K/BB_steamer', 'HR/9_steamer', 'K%_steamer',
       'BB%_steamer', 'K-BB%_steamer', 'AVG_steamer', 'WHIP_steamer',
       'BABIP_steamer', 'LOB%_steamer', 'GB%_steamer', 'HR/FB_steamer',
       'FIP_steamer', 'WAR_steamer', 'RA9-WAR_steamer','PlayerId_steamer', 'FIPdifference']]

zipsSteamer["IP_zips"] = pd.to_numeric(zipsSteamer["IP_zips"], errors='coerce')
zipsSteamer["IP_steamer"] = pd.to_numeric(zipsSteamer["IP_steamer"], errors='coerce')
minInnings = 80
zipsSteamer = zipsSteamer.query(f"IP_zips > {minInnings} and IP_steamer > {minInnings}")


# Plotting the y=x line
min_val = min(min((zipsSteamer.WAR_steamer/zipsSteamer.IP_steamer)*162), min((zipsSteamer.WAR_zips/zipsSteamer.IP_zips)*162))
max_val = max(max((zipsSteamer.WAR_steamer/zipsSteamer.IP_steamer)*162), max((zipsSteamer.WAR_zips/zipsSteamer.IP_zips)*162))
plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--')


plt.scatter((zipsSteamer.WAR_steamer/zipsSteamer.IP_steamer)*162, (zipsSteamer.WAR_zips/zipsSteamer.IP_zips)*162)

cursor = mplcursors.cursor(hover=True)
cursor.connect("add", lambda sel: sel.annotation.set_text(
   f"Pitcher: {zipsSteamer['Name_zips'].iloc[sel.index]}\nRank, WAR, IP (ZiPS): {zipsSteamer.zips_index.iloc[sel.index]}, {zipsSteamer.WAR_zips.iloc[sel.index]}, {zipsSteamer.IP_zips.iloc[sel.index]}\nRank, WAR, IP (Steamer): {zipsSteamer.steamer_index.iloc[sel.index]}, {zipsSteamer.WAR_steamer.iloc[sel.index]}, {zipsSteamer.IP_steamer.iloc[sel.index]}"
))

plt.ylabel("2024 ZiPS WAR/162 Innings Projection")
plt.xlabel("2024 Steamer WAR/162 Innings Projection")
plt.title("2024 WAR/162 Innings Projection, ZiPS vs Steamer")
corr = (zipsSteamer.WAR_steamer/zipsSteamer.IP_steamer*162).corr(zipsSteamer.WAR_zips/zipsSteamer.IP_zips*162)
plt.suptitle(f"Correlation: {corr}, Minimum Projected Innings: {minInnings}")
plt.gca().set_aspect('equal', adjustable='box')
plt.show()