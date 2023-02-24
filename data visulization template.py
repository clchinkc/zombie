
"""
import matplotlib.pyplot as plt
from matplotlib import markers

years = [1950,1960,1965,1970,1975,1980,
        1985,1990,1995,2000,2005,
        2010,2015]
pops = [2.5,2.7,3,3.3,3.6,4.0,
        4.4,4.8,5.3,6.1,6.5,6.9,7.3]
deaths = [1.2,1.7,1.8,2.2,2.5,
        2.7,2.9,3,3.1,3.2,3.5,3.6,4]

pops_line = plt.plot(years,pops, color=(255/255,100/255,100/255))
deaths_line = plt.plot(years,deaths, '--', color=(100/255,100/255,255/255))

plt.title("Population Growth") # title
plt.ylabel("Population in billions") # y label
plt.xlabel("Population growth by year") # x label

plt.setp(plt.gca().get_xticklabels(), horizontalalignment='right')
plt.setp(plt.gca().get_yticklabels(), horizontalalignment='right')
plt.setp(pops_line, marker='o', markersize=5, markerfacecolor='red')
plt.setp(deaths_line, marker='o', markersize=5, markerfacecolor='blue')

plt.grid(True)

plt.show()
"""

"""
import matplotlib.pyplot as plt
import numpy as np

x = np.arange(0, 1.0, 0.01)
y1 = np.sin(4*np.pi*x)
y2 = np.sin(2*np.pi*x)
lines = plt.plot(x, y1, x, y2)
l1, l2 = lines
plt.setp(lines, linestyle='--')      
plt.show()
"""


# plot_web_api_realtime.py
"""
A live auto-updating plot of random numbers pulled from a web API
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
    ax.set_ylim([0,255])
    plt.xticks(rotation=45, ha='right')
    plt.subplots_adjust(bottom=0.20)
    ax.set_title('Plot of random numbers from https://qrng.anu.edu.au')
    ax.set_xlabel('Date Time (hour:minute:second)')
    ax.set_ylabel('Random Number')

# Set up plot to call animate() function every 1000 milliseconds
ani = animation.FuncAnimation(fig, animate, fargs=(xs,ys), interval=1000)

plt.show()