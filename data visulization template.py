"""
# Plotting a line graph
import matplotlib.pyplot as plt
from matplotlib import markers

years = [1950,1960,1965,1970,1975,1980,
        1985,1990,1995,2000,2005,
        2010,2015]
pops = [2.5,2.7,3,3.3,3.6,4.0,
        4.4,4.8,5.3,6.1,6.5,6.9,7.3]
deaths = [1.2,1.7,1.8,2.2,2.5,
        2.7,2.9,3,3.1,3.2,3.5,3.6,4]

pops_line = plt.plot(years,pops, color=(255/255,100/255,100/255), alpha=0.5, linewidth=3)
deaths_line = plt.plot(years,deaths, '--', color=(100/255,100/255,255/255), alpha=0.5, linewidth=2)

plt.title("Population Growth") # title
plt.ylabel("Population in billions") # y label
plt.xlabel("Population growth by year") # x label

plt.setp(plt.gca().get_xticklabels(), horizontalalignment='right')
plt.setp(plt.gca().get_yticklabels(), horizontalalignment='right')
plt.setp(pops_line, marker='o', markersize=5, markerfacecolor='red')
plt.setp(deaths_line, marker='o', markersize=5, markerfacecolor='blue')

plt.legend(['Population','Deaths'], loc='upper left')
plt.grid(True)

plt.tight_layout()
plt.show()
"""

"""
# Plotting a bar graph

import matplotlib.pyplot as plt
import numpy as np

categories =  ['A','B','C','D','E']
sales = np.random.randint(3000,9000,5)
costs = np.random.randint(1000,7000,5)
errors = np.random.randint(100,500,5)

bar_width = 0.35
index = np.arange(len(categories))

plt.bar(index, sales, bar_width, yerr = errors, label='Sales', tick_label=categories, color = ['red','blue','green','yellow','purple'], edgecolor='black', linewidth=2)
plt.bar(index + bar_width, costs, bar_width, yerr = errors, label='Cost', tick_label=categories, color = ['red','blue','green','yellow','purple'], edgecolor='black', linewidth=2)

plt.xlim(-0.5, len(categories))
plt.ylim(0, 10000)

# Plot the data
plt.xlabel('Categories')
plt.ylabel('Sales and Cost')
plt.title('Sales and Cost by Category')

# Adding the data lables to the columns
for i, v in enumerate(sales):
    plt.text(i, v+errors[i], str(v), color = 'red', ha = 'center', va = 'bottom')
for i, v in enumerate(costs):
    plt.text(i + bar_width, v+errors[i], str(v), color = 'blue', ha = 'center', va = 'bottom')
    
plt.legend(ncol=2, loc='upper right', bbox_to_anchor=(1,1), columnspacing=1, labelspacing=0.5, handletextpad=0.5, handlelength=1.5, shadow=True)

plt.tight_layout()
plt.show()
"""


# plot_web_api_realtime.py
"""
A live auto-updating plot of random numbers pulled from a web API
"""
"""
import datetime as dt
import time

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import requests

url = "https://qrng.anu.edu.au/API/jsonI.php?length=1&type=uint8"

# function to pull out a float from the requests response object
def pull_float(response):
    jsonr = response.json()
    strr = jsonr["data"][0]
    if strr:
        fltr = round(float(strr), 2)
        return fltr
    else:
        return None


# Create figure for plotting
fig, ax = plt.subplots()
xs = []
ys = []

def animate(i, xs:list, ys:list):
    # grab the data from thingspeak.com
    response = requests.get(url)
    flt = pull_float(response)
    # Add x and y to lists
    xs.append(dt.datetime.now().strftime('%H:%M:%S'))
    ys.append(flt)
    # Limit x and y lists to 10 items
    xs = xs[-10:]
    ys = ys[-10:]
    # Draw x and y lists
    ax.clear()
    ax.plot(xs, ys)
    # Format plot
    ax.set_ylim(0, 255)
    plt.xticks(rotation=45, ha='right')
    plt.subplots_adjust(bottom=0.20)
    ax.set_title('Plot of random numbers from https://qrng.anu.edu.au')
    ax.set_xlabel('Date Time (hour:minute:second)')
    ax.set_ylabel('Random Number')

# Set up plot to call animate() function every 1000 milliseconds
ani = animation.FuncAnimation(fig, animate, fargs=(xs,ys), interval=1000)

plt.show()
"""

"""
# seaborn

import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="white", palette="muted", color_codes=True)

# Load the example dataset
iris = sns.load_dataset("iris")

# Set up the matplotlib figure
fig, axes = plt.subplots(2, 2)

# Plot suitable plots
sns.heatmap(iris.corr(numeric_only=True), annot=True, ax=axes[0, 0])
sns.histplot(iris.sepal_length, kde=False, color="b", ax=axes[0, 1])
sns.kdeplot(iris.sepal_width, fill=True, color="r", ax=axes[1, 0])
sns.rugplot(iris.sepal_width, color="g", ax=axes[1, 1], height=0.2)

plt.tight_layout()
plt.show()

# displot is a figure level function that combines histplot and kdeplot
# jointplot is a figure level function that combines scatterplot and histplot
# pairplot is a figure level function that combines scatterplot and histplot
# facetgrid is a figure level function
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set(style="ticks", palette="muted", color_codes=True)

plt.figure(figsize=(12, 6))

# Load the example planets dataset
planets = sns.load_dataset("planets")

# Make a rank of distances
ranks = planets.groupby("method")["distance"].mean().fillna(0).sort_values()[::-1].index

# Plot the orbital period with horizontal boxes
ax = sns.boxplot(x="distance", y="method", data=planets, whis=30, color="c", order=ranks)

# Add in points to show each observation
sns.stripplot(x="distance", y="method", data=planets, jitter=True, size=3, color=".3", linewidth=0, order=ranks)

# Make the quantitative axis logarithmic
ax.set_xscale("log")
sns.despine(trim=True)

plt.tight_layout()
plt.show()

# Probability plot
# Spaghetti plot
# Biplot
# Residual plot
# Pairplot
# Violin plot
# Boxen plot
# Swarm plot
# Catplot

