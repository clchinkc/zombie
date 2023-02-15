# tutorial for matplotlib

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm

"""# line plot using colormap

# data
x = np.linspace(0, 10, 100)
y = np.sin(x)

# plot
cmap = cm.get_cmap('jet')(np.linspace(0, 1, len(x)))
plt.scatter(x, y, c=cmap, s=100, alpha=0.5)
plt.show()"""

"""# grouped bar chart

# data
x = np.arange(5)
y1 = np.random.randint(1, 10, size=5)
y2 = np.random.randint(1, 10, size=5)
y3 = np.random.randint(1, 10, size=5)

# plot
width = 0.25
plt.bar(x, y1, color='r', width=width, label='y1')
plt.bar(x+width, y2, color='g', width=width, label='y2')
plt.bar(x+2*width, y3, color='b', width=width, label='y3')
plt.xticks(x + width, ('x1', 'x2', 'x3', 'x4', 'x5'))
plt.ylabel('y')
plt.legend(bbox_to_anchor=(1,1), loc='upper left')
plt.show()"""

"""# stacked bar chart

# data
x = np.arange(5)
y1 = np.random.randint(1, 10, size=5)
y2 = np.random.randint(1, 10, size=5)
y3 = np.random.randint(1, 10, size=5)

# plot
width = 0.25
plt.bar(x, y1, color='r', width=width, label='y1')
plt.bar(x, y2, color='g', width=width, label='y2', bottom=y1)
plt.bar(x, y3, color='b', width=width, label='y3', bottom=y1+y2)
plt.xticks(x, ('x1', 'x2', 'x3', 'x4', 'x5'))
plt.ylabel('y')
plt.legend(bbox_to_anchor=(1,1), loc='upper left')
plt.show()"""

"""# histogram

# data
x = np.random.randn(100000)

# plot
plt.hist(x, bins=200)
plt.show()"""

"""# pie chart

# data
x = np.random.randint(1, 10, size=5)

explode = np.zeros(5)
explode[3] = 0.2

# plot
plt.pie(x, labels=('x1', 'x2', 'x3', 'x4', 'x5'), autopct='%1.2f%%', explode=explode)
plt.title('pie chart')
plt.show()"""

"""# sunburst chart

# data
x = np.random.randint(1, 10, size=5)
y = np.random.randint(1, 10, size=5)

# plot
fig, ax = plt.subplots()
ax.pie(x, labels=['x1', 'x2', 'x3', 'x4', 'x5'], autopct='%1.2f%%', radius=1, labeldistance=0.6, pctdistance=0.8, wedgeprops=dict(width=0.4, edgecolor='w'))
ax.pie(y, labels=['y1', 'y2', 'y3', 'y4', 'y5'], autopct='%1.2f%%', radius=0.6, labeldistance=0.3, pctdistance=0.5)
ax.set(title='sunburst chart')
plt.show()"""

import plotly.express as px

# data
x = np.random.randint(1, 10, size=5)

# plot
fig = px.pie(x, values=x, names=['x1', 'x2', 'x3', 'x4', 'x5'], title='pie chart', hole=0.1)
fig.show()

import plotly.graph_objects as go

pull = np.zeros(5)
pull[3] = 0.2

fig = go.Figure(data=[go.Pie(labels=['x1', 'x2', 'x3', 'x4', 'x5'], values=x, pull=pull, hole=0.1)])
fig.show()

# sunburst plot in plotly
# https://plotly.com/python/sunburst-charts/
