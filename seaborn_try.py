
from textwrap import fill

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# seaborn has five preset themes: darkgrid, whitegrid, dark, white, and ticks
# sns.set_style("whitegrid")
# palette: deep, muted, pastel, bright, dark, and colorblind
# color_codes: True or False, true to use color codes for the colors in the palette
sns.set(style="whitegrid", palette="muted", color_codes=True)

"""
df_iris = sns.load_dataset('iris')
fig, axes = plt.subplots(1,2)
# distplot: plot a univariate distribution of observations
sns.histplot(df_iris['petal_length'], ax = axes[0], kde = True) # kde: kernel density estimation
# kdeplot: plot univariate or bivariate distributions using kernel density estimation
sns.kdeplot(df_iris['petal_length'], ax = axes[1], fill=True) # fill: fill the area under the curve                      
plt.show()
"""

"""
rs = np.random.RandomState(10)  
d = rs.normal(size=100)  
f, axes = plt.subplots(2, 2, figsize=(7, 7), sharex=True)  
sns.histplot(d, kde=False, color="b", ax=axes[0, 0])  
sns.rugplot(d, color="r", ax=axes[0, 0])
sns.kdeplot(d, fill=True, cumulative=True, color="g", ax=axes[1, 0])  
# fit a parametric distribution to a univariate dataset
# sns.distplot(d, kde=False, fit=stats.gamma, ax=axes[0, 1])
# latest version of this code
sns.histplot(d, stat="density", kde=True, color="g", ax=axes[1, 1])
plt.tight_layout()
plt.show()
"""
"""
from scipy import stats

d = np.random.normal(size=500) * 0.1
mu, std = stats.norm.fit(d)

# Plot the histogram.
plt.hist(d, bins=25, density=True, alpha=0.6, color='g')

# Plot the PDF.
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = stats.norm.pdf(x, mu, std)
plt.plot(x, p, 'k', linewidth=2)
title = "Fit results: mu = %.2f,  std = %.2f" % (mu, std)
plt.title(title)

plt.show()
"""
"""
iris = sns.load_dataset("iris")
sns.stripplot(x="species", y="petal_length", data=iris, jitter=True) # jitter: amount of jitter (only along the categorical axis)
plt.show()
"""
"""
iris = sns.load_dataset("iris")
sns.swarmplot(x="species", y="petal_length", data=iris)
plt.show()
"""
"""
iris = sns.load_dataset("iris")
sns.violinplot(x="petal_length", data=iris)
plt.show()
"""
"""
tips = sns.load_dataset("tips")
sns.set(style="ticks") # ticks: axis ticks are visible
sns.boxplot(x="day", y="total_bill", hue="sex", data=tips, palette="PRGn") # palette: color palette
plt.show()  
"""
"""
titanic = sns.load_dataset("titanic")
plt.subplot(1, 2, 1)
sns.countplot(x="class", hue="who", data=titanic)
plt.subplot(1, 2, 2)
sns.countplot(x="who", data=titanic, facecolor=(0, 0, 0, 0), linewidth=5, edgecolor=sns.color_palette("dark", 3))
plt.show()
"""
"""
import pandas as pd

rs = np.random.RandomState(10)  
mean, cov = [0, 1], [(1, .5), (.5, 1)]  
data = rs.multivariate_normal(mean, cov, 200)  
df = pd.DataFrame(data, columns=["x", "y"])  
# jointplot: plot a relationship between two variables
sns.jointplot(x="x", y="y", data=df)
sns.jointplot(x="x", y="y", data=df, kind="hex") # hex: hexbin plot
sns.jointplot(x="x", y="y", data=df, kind="kde") # kde: kernel density estimation
plt.show()
"""
"""
tips = sns.load_dataset("tips")
sns.jointplot(x="total_bill", y="tip", data=tips, kind='reg')       
plt.show() 
"""

"""
data = sns.load_dataset('iris')
data = data.corr(numeric_only=True)
sns.heatmap(data)  
plt.show()  
"""
"""
data = sns.load_dataset('iris')
sns.set()
sns.pairplot(data,hue="species",palette="husl") # hue: variable that defines subsets of the data, which will be drawn on separate facets in the grid, vars: variables to use
plt.show()  
"""
"""
tips = sns.load_dataset("tips")
g = sns.FacetGrid(tips, col="time",  row="smoker")  
g = g.map(plt.hist, "total_bill",  color="r")  
plt.show()  
"""
"""
tips = sns.load_dataset("tips")
sns.relplot(x="total_bill", y="tip", col="time", hue="smoker", style="smoker", size="size", data=tips)
sns.lmplot(x="total_bill", y="tip", col="time", hue="smoker", data=tips)
plt.show()
"""
"""
tips = sns.load_dataset("tips")
sns.lmplot(x="total_bill", y="tip", col="time", hue="smoker", data=tips)
plt.show()
"""
"""
dots = sns.load_dataset("dots")
sns.relplot(x="time", y="firing_rate", col="align", hue="choice", style="choice", size="coherence", facet_kws=dict(sharex=False), kind="line", legend="full", data=dots)
plt.show()
"""
"""
fmri = sns.load_dataset("fmri")
sns.relplot(x="timepoint", y="signal", hue="event", style="event", col="region", kind="line", data=fmri)
plt.show()
"""

tips = sns.load_dataset("tips")
sns.relplot(x="total_bill", y="tip", data=tips)
sns.relplot(x="total_bill", y="tip", data=tips, kind="scatter") # kind: {‘scatter’, 'line'}
plt.show()

"""
tips = sns.load_dataset("tips")
sns.catplot(x="day", y="total_bill", hue="smoker", col="time", data=tips, kind="violin") # kind: {point, strip, swarm, bar, count, box, violin}
sns.catplot(x="day", y="total_bill", hue="smoker", col="time", data=tips, kind="violin", split=True, inner=None)
plt.show()
"""


# https://blog.csdn.net/pythonxiaopeng/article/details/109642444?utm_medium=distribute.pc_relevant.none-task-blog-2~default~baidujs_baidulandingword~default-5-109642444-blog-73432410.pc_relevant_3mothn_strategy_and_data_recovery&spm=1001.2101.3001.4242.4&utm_relevant_index=7