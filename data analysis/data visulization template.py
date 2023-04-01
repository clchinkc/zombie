

"""
# Plotting a line graph
import matplotlib.pyplot as plt

years = [1950,1960,1965,1970,1975,1980,
        1985,1990,1995,2000,2005,
        2010,2015]
pops = [2.5,2.7,3,3.3,3.6,4.0,
        4.4,4.8,5.3,6.1,6.5,6.9,7.3]
deaths = [1.2,1.7,1.8,2.2,2.5,
        2.7,2.9,3,3.1,3.2,3.5,3.6,4]
birth = [4.5,4.7,4.8,5.0,5.2,
        5.4,5.6,5.8,6.0,6.2,6.4,6.6,7]

fig, ax1 = plt.subplots()

# Population plot
pops_line = ax1.plot(years, pops, color=(255/255, 100/255, 100/255), alpha=0.5, linewidth=3)
ax1.set_xlabel('Year')
ax1.set_ylabel('Population in billions', color=(255/255, 100/255, 100/255))
ax1.tick_params(axis='y', labelcolor=(255/255, 100/255, 100/255))
ax1.legend(['Population'], loc='upper left')

# Birth and death rate plot
ax2 = ax1.twinx()
birth_line = ax2.plot(years, birth, color='green', alpha=0.5, linewidth=3)
death_line = ax2.plot(years, deaths, color='blue', alpha=0.5, linewidth=3)
ax2.set_ylabel('Birth and Death Rates', color='black')
ax2.tick_params(axis='y', labelcolor='black')
ax2.legend(['Birth Rate', 'Death Rate'], loc='lower right')

# Inset plot
left, bottom, width, height = [0.12, 0.7, 0.2, 0.2]
axins = ax2.inset_axes([left, bottom, width, height])
diff = [birth[i]-deaths[i] for i in range(len(birth))]
diff_line = axins.plot(years, diff, color='purple', alpha=0.5, linewidth=1)
axins.set_ylabel('Birth - Death', color='black')
axins.tick_params(axis='y', labelcolor='black')

# Other settings
plt.setp(plt.gca().get_xticklabels(), horizontalalignment='right')
plt.setp(plt.gca().get_yticklabels(), horizontalalignment='right')
plt.setp(pops_line, marker='o', markersize=5, markerfacecolor='red')
plt.setp(birth_line, marker='o', markersize=5, markerfacecolor='green')
plt.setp(death_line, marker='o', markersize=5, markerfacecolor='blue')
plt.setp(diff_line, marker='o', markersize=1, markerfacecolor='purple')

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


# seaborn
"""

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
"""

# Probability plot
# Spaghetti plot
# Biplot
# Residual plot
# Pairplot
# Violin plot
# Boxen plot
# Swarm plot
# Catplot


# bayesian
"""
import math

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation


def beta_pdf(x, a, b):
    return (x**(a-1) * (1-x)**(b-1) * math.gamma(a + b)
            / (math.gamma(a) * math.gamma(b)))


class UpdateDist:
    def __init__(self, ax, prob=0.5):
        self.success = 0
        self.prob = prob
        self.line, = ax.plot([], [], 'k-')
        self.x = np.linspace(0, 1, 200)
        self.ax = ax

        # Set up plot parameters
        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(0, 10)
        self.ax.grid(True)

        # This vertical line represents the theoretical value, to
        # which the plotted distribution should converge.
        self.ax.axvline(prob, linestyle='--', color='black')

    def __call__(self, i):
        # This way the plot can continuously run and we just keep
        # watching new realizations of the process
        if i == 0:
            self.success = 0
            self.line.set_data([], [])
            return self.line,

        # Choose success based on exceed a threshold with a uniform pick
        if np.random.rand() < self.prob:
            self.success += 1
        y = beta_pdf(self.x, self.success + 1, (i - self.success) + 1)
        self.line.set_data(self.x, y)
        return self.line,

# Fixing random state for reproducibility
np.random.seed(19680801)


fig, ax = plt.subplots()
ud = UpdateDist(ax, prob=0.7)
anim = FuncAnimation(fig, ud, frames=100, interval=100, blit=True)
plt.show()
"""


# 3D random walk
"""
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

# Fixing random state for reproducibility
np.random.seed(19680801)


def random_walk(num_steps, max_step=0.05):
    # Return a 3D random walk as (num_steps, 3) array.
    start_pos = np.random.random(3)
    steps = np.random.uniform(-max_step, max_step, size=(num_steps, 3))
    walk = start_pos + np.cumsum(steps, axis=0)
    return walk

def update_lines(num, walks, lines):
    for line, walk in zip(lines, walks):
        # NOTE: there is no .set_data() for 3 dim data...
        line.set_data(walk[:num, :2].T)
        line.set_3d_properties(walk[:num, 2])
    return lines

# Data: 40 random walks as (num_steps, 3) arrays
num_steps = 30
walks = [random_walk(num_steps) for index in range(40)]

# Attaching 3D axis to the figure
fig = plt.figure()
ax = fig.add_subplot(projection="3d")

# Create lines initially without data
lines = [ax.plot([], [], [])[0] for _ in walks]

# Setting the axes properties
ax.set(xlim3d=(0, 1), xlabel='X')
ax.set(ylim3d=(0, 1), ylabel='Y')
ax.set(zlim3d=(0, 1), zlabel='Z')

# Creating the Animation object
ani = animation.FuncAnimation(
    fig, update_lines, num_steps, fargs=(walks, lines), interval=100)

plt.show()
"""


# Oscilloscope
"""
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


class Scope:
    def __init__(self, ax, maxt=2, dt=0.02):
        self.ax = ax
        self.dt = dt
        self.maxt = maxt
        self.tdata = [0]
        self.ydata = [0]
        self.line = Line2D(self.tdata, self.ydata)
        self.ax.add_line(self.line)
        self.ax.set_ylim(-.1, 1.1)
        self.ax.set_xlim(0, self.maxt)

    def update(self, y):
        lastt = self.tdata[-1]
        if lastt >= self.tdata[0] + self.maxt:  # reset the arrays
            self.tdata = [self.tdata[-1]]
            self.ydata = [self.ydata[-1]]
            self.ax.set_xlim(self.tdata[0], self.tdata[0] + self.maxt)
            self.ax.figure.canvas.draw()

        # This slightly more complex calculation avoids floating-point issues
        # from just repeatedly adding `self.dt` to the previous value.
        t = self.tdata[0] + len(self.tdata) * self.dt

        self.tdata.append(t)
        self.ydata.append(y)
        self.line.set_data(self.tdata, self.ydata)
        return self.line,


def emitter(p=0.1):
    # Return a random value in [0, 1) with probability p, else 0.
    while True:
        v = np.random.rand()
        if v > p:
            yield 0.
        else:
            yield np.random.rand()


# Fixing random state for reproducibility
np.random.seed(19680801 // 10)


fig, ax = plt.subplots()
scope = Scope(ax)

# pass a generator in "emitter" to produce data for the update func
ani = animation.FuncAnimation(fig, scope.update, emitter, interval=50,
                              blit=True, save_count=100)

plt.show()
"""


# Pendulum
"""
from collections import deque

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from numpy import cos, sin

G = 9.8  # acceleration due to gravity, in m/s^2
L1 = 1.0  # length of pendulum 1 in m
L2 = 1.0  # length of pendulum 2 in m
L = L1 + L2  # maximal length of the combined pendulum
M1 = 1.0  # mass of pendulum 1 in kg
M2 = 1.0  # mass of pendulum 2 in kg
t_stop = 2.5  # how many seconds to simulate
history_len = 500  # how many trajectory points to display


def derivs(t, state):
    dydx = np.zeros_like(state)

    dydx[0] = state[1]

    delta = state[2] - state[0]
    den1 = (M1+M2) * L1 - M2 * L1 * cos(delta) * cos(delta)
    dydx[1] = ((M2 * L1 * state[1] * state[1] * sin(delta) * cos(delta)
                + M2 * G * sin(state[2]) * cos(delta)
                + M2 * L2 * state[3] * state[3] * sin(delta)
                - (M1+M2) * G * sin(state[0]))
               / den1)

    dydx[2] = state[3]

    den2 = (L2/L1) * den1
    dydx[3] = ((- M2 * L2 * state[3] * state[3] * sin(delta) * cos(delta)
                + (M1+M2) * G * sin(state[0]) * cos(delta)
                - (M1+M2) * L1 * state[1] * state[1] * sin(delta)
                - (M1+M2) * G * sin(state[2]))
               / den2)

    return dydx

# create a time array from 0..t_stop sampled at 0.02 second steps
dt = 0.01
t = np.arange(0, t_stop, dt)

# th1 and th2 are the initial angles (degrees)
# w10 and w20 are the initial angular velocities (degrees per second)
th1 = 120.0
w1 = 0.0
th2 = -10.0
w2 = 0.0

# initial state
state = np.radians([th1, w1, th2, w2])

# integrate the ODE using Euler's method
y = np.empty((len(t), 4))
y[0] = state
for i in range(1, len(t)):
    y[i] = y[i - 1] + derivs(t[i - 1], y[i - 1]) * dt

# A more accurate estimate could be obtained e.g. using scipy:
#
#   y = scipy.integrate.solve_ivp(derivs, t[[0, -1]], state, t_eval=t).y.T

x1 = L1*sin(y[:, 0])
y1 = -L1*cos(y[:, 0])

x2 = L2*sin(y[:, 2]) + x1
y2 = -L2*cos(y[:, 2]) + y1

fig = plt.figure(figsize=(5, 4))
ax = fig.add_subplot(autoscale_on=False, xlim=(-L, L), ylim=(-L, 1.))
ax.set_aspect('equal')
ax.grid()

line, = ax.plot([], [], 'o-', lw=2)
trace, = ax.plot([], [], '.-', lw=1, ms=2)
time_template = 'time = %.1fs'
time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)
history_x, history_y = deque(maxlen=history_len), deque(maxlen=history_len)


def animate(i):
    thisx = [0, x1[i], x2[i]]
    thisy = [0, y1[i], y2[i]]

    if i == 0:
        history_x.clear()
        history_y.clear()

    history_x.appendleft(thisx[2])
    history_y.appendleft(thisy[2])

    line.set_data(thisx, thisy)
    trace.set_data(history_x, history_y)
    time_text.set_text(time_template % (i*dt))
    return line, trace, time_text


ani = animation.FuncAnimation(
    fig, animate, len(y), interval=dt*1000, blit=True)
plt.show()
"""


# offset piston motion
"""
import matplotlib.animation as animation
import matplotlib.pyplot as plt

# import the necessary packages
import numpy as np
from numpy import cos, pi, sin, sqrt

# input parameters
r = 1.0  # crank radius
l = 4.0  # connecting rod length
d = 0.5  # offset distance
rot_num = 6 # number of crank rotations
increment = 0.1 # angle incremement

# create the angle array, where the last angle is the number of rotations*2*pi
angle_minus_last = np.arange(0,rot_num*2*pi,increment)
angles = np.append(angle_minus_last, rot_num*2*pi)

X1=np.zeros(len(angles)) # crank x-positions: Point 1
Y1=np.zeros(len(angles)) # crank y-positions: Point 1
X2=np.zeros(len(angles)) # connecting rod x-positions: Point 2
Y2=np.zeros(len(angles)) # connecting rod y-positions: Point 2

# calculate the crank and connecting rod positions for each angle
for index,theta in enumerate(angles, start=0):
    x1 = r*cos(theta) # x-cooridnate of the crank: Point 1
    y1 = r*sin(theta) # y-cooridnate of the crank: Point 1
    x2 = d # x-coordinate of the rod: Point 2
    y2 = r*sin(theta) + sqrt(l**2 - (r*cos(theta)-d)**2) # y-coordinate of the rod: Point 2
    X1[index]=x1 # crankshaft x-position
    Y1[index]=y1 # crankshaft y-position
    X2[index]=x2 # connecting rod x-position
    Y2[index]=y2 # connecting rod y-position

# set up the figure and subplot
fig = plt.figure()
ax = fig.add_subplot(111, aspect='equal', autoscale_on=False, xlim=(-4,4), ylim=(-2,6))
ax.grid()
ax.set_title('Offset Piston Motion Animation')
ax.axes.xaxis.set_ticklabels([])
ax.axes.yaxis.set_ticklabels([])
line, = ax.plot([], [], 'o-', lw=5, color='#de2d26')


# initialization function
def init():
    line.set_data([], [])
    return line,


# animation function
def animate(i):
    x_points = [0, X1[i], X2[i]]
    y_points = [0, Y1[i], Y2[i]]
    line.set_data(x_points, y_points)
    return line,


# call the animation
ani = animation.FuncAnimation(fig, animate, init_func=init, frames=len(X1), interval=40, blit=True, repeat=False)
## to save animation, uncomment the line below:
## ani.save('offset_piston_motion_animation.mp4', fps=30, extra_args=['-vcodec', 'libx264'])

# show the animation
plt.show()
"""


# rocket motion
"""
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from numpy import absolute, arccos, arctan, cos, pi, sign, sin, sqrt

# input parameters
r = 0.395  # crank radius
l = 4.27  # connecting rod length
rr = 1.15 # rocker radius
d = 4.03  # center-to-center distance
rot_num = 6  # number of crank rotations
increment = 0.1  # angle incremement
over = 1  # if over = 1 --> mechanism is on top, If over = -1, mechanism on bottom
s = over / absolute(over)

# create the angle array, where the last angle is the number of rotations*2*pi
angle_minus_last = np.arange(0, rot_num * 2 * pi, increment)
R_Angles = np.append(angle_minus_last, rot_num * 2 * pi)

# coordinates of the crank center point : Point 1
x1 = 0
y1 = 0

# Coordinates of the rocker center point: Point 4
x4 = d
y4 = 0

X2 = np.zeros(len(R_Angles))  # array of crank x-positions: Point 2
Y2 = np.zeros(len(R_Angles))  # array of crank y-positions: Point 2
RR_Angle = np.zeros(len(R_Angles))  # array of rocker arm angles
X3 = np.zeros(len(R_Angles))  # array of rocker x-positions: Point 3
Y3 = np.zeros(len(R_Angles))  # array of rocker y-positions: Point 3

# find the crank and connecting rod positions for each angle
for index, R_Angle in enumerate(R_Angles, start=0):
    theta1 = R_Angle
    x2 = r * cos(theta1)  # x-cooridnate of the crank: Point 2
    y2 = r * sin(theta1)  # y-cooridnate of the crank: Point 2
    e = sqrt((x2 - d) ** 2 + (y2 ** 2))
    phi2 = arccos((e ** 2 + rr ** 2 - l ** 2) / (2 * e * rr))
    phi1 = arctan(y2 / (x2 - d)) + (1 - sign(x2 - d)) * pi / 2
    theta3 = phi1 - s * phi2
    RR_Angle[index] = theta3
    x3 = rr * cos(theta3) + d
    # x cooridnate of the rocker moving point: Point 3
    y3 = rr * sin(theta3)
    # y cooridnate of the rocker moving point: Point 3

    theta2 = arctan((y3 - y2) / (x3 - x2)) + (1 - sign(x3 - x2)) * pi / 2

    X2[index] = x2  # grab the crankshaft x-position
    Y2[index] = y2  # grab the crankshaft y-position
    X3[index] = x3  # grab the connecting rod x-position
    Y3[index] = y3  # grab the connecting rod y-position
    
# set up the figure and subplot
fig = plt.figure()
ax = fig.add_subplot(
    111, aspect="equal", autoscale_on=False, xlim=(-2, 6), ylim=(-2, 3)
)

# add grid lines, title and take out the axis tick labels
ax.grid(alpha=0.5)
ax.set_title("Crank and Rocker Motion")
ax.set_xticklabels([])
ax.set_yticklabels([])
(line, ) = ax.plot(
    [], [], "o-", lw=5, color="#2b8cbe"
)  # color from: http://colorbrewer2.org/

# initialization function
def init():
    line.set_data([], [])
    return (line,)


# animation function
def animate(i):
    x_points = [x1, X2[i], X3[i], x4]
    y_points = [y1, Y2[i], Y3[i], y4]

    line.set_data(x_points, y_points)
    return (line,)

# call the animation
ani = animation.FuncAnimation(
    fig, animate, init_func=init, frames=len(X2), interval=40, blit=True, repeat=False
)
## to save animation, uncomment the line below. Ensure ffmpeg is installed:
## ani.save('crank_and_rocker_motion_animation.mp4', fps=30, extra_args=['-vcodec', 'libx264'])

# show the animation
plt.show()
"""

# Dynamical Systems with Python: Lorenz System

import matplotlib.pyplot as plt

# Import Required Libraries:
import numpy as np
from scipy.integrate import odeint

# System Coefficients:
rho = 28.0
sigma = 10.0
beta = 8.0 / 3.0

def f(state, t):
    
    # Unpack the state vector:
    x, y, z = state  
    
    # Define the system here:
    xdot=sigma * (y - x)
    ydot=x * (rho - z) - y
    zdot=x * y - beta * z 
    
    # Return derivatives:
    return xdot, ydot, zdot  

# Initial condition:
state0 = [1.0, 1.0, 1.0]

# Define time intervals:
t = np.arange(0.0, 100.0, 0.01)

# Calculate states:
states = odeint(f, state0, t)

# Unpack the states:
x=states[:,0]
y=states[:,1]
z=states[:,2]

# Draw the 3D plot of the states:

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

ax.plot(x, y, z,'b')
ax.plot(state0[0],state0[1],state0[2],'*')

ax.set_xlim([-25,25])
ax.set_ylim([-25,25])
ax.set_zlim([0,50])

ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')

plt.draw()
plt.show()
