import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mplcursors

zips = pd.read_csv("/Users/leofeingold/Desktop/fangraphs/hittingProjections/hittingZipsDC.csv").rename(columns = lambda x: x.lower())
zips = zips.rename(columns = lambda x: x + "_zips")
steamer = pd.read_csv("/Users/leofeingold/Desktop/fangraphs/hittingProjections/hittingSteamer.csv").rename(columns = lambda x: x.lower())
steamer = steamer.rename(columns = lambda x: x + "_steamer")
steamer.dropna()
zips.dropna()

#print(steamer.loc[steamer['name_steamer'] == 'Max Muncy'].babip_steamer)

zipsSteamer = pd.merge(zips, steamer, left_on = 'playerid_zips', right_on = 'playerid_steamer')

xItem = "wrc+_zips"
yItem = "wrc+_steamer"
xCol = zipsSteamer[xItem]
yCol = zipsSteamer[yItem]
corr = xCol.corr(yCol)


# Plotting the y=x line
min_val = min(min(xCol), min(yCol))
max_val = max(max(xCol), max(yCol))
plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--')

plt.scatter(xCol, yCol)

cursor = mplcursors.cursor(hover=True)
cursor.connect("add", lambda sel: sel.annotation.set_text(
   f"Name: {zipsSteamer['name_zips'].iloc[sel.index]},\n"
   f"wRC+ Difference (ZiPS, Steamer): {abs(zipsSteamer['wrc+_zips'].iloc[sel.index] - zipsSteamer['wrc+_steamer'].iloc[sel.index])}"
))


plt.xlabel(f"{xItem}")
plt.ylabel(f"{yItem}")
plt.suptitle(f"{xItem} vs {yItem}")
plt.title(f"Correlation: {corr}")
plt.gca().set_aspect('equal', adjustable='box')
plt.show()
