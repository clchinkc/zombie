# tutorial for animating a scatter plot
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

"""
# create a figure
fig = plt.figure()

# create a subplot
ax1 = fig.add_subplot(1,1,1)

# create a function that will do the plotting, where curr is the current frame
def animate(curr):
    # check if animation is at the last frame, and if so, stop the animation a
    if curr == 100:
        a.event_source.stop()
    # clear the axes and redraw the plot for the next frame
    ax1.clear()
    ax1.plot(x[:curr], y[:curr])
    
# create a variable that will contain the animation
a = animation.FuncAnimation(fig, animate, interval=100)

# create data
x = np.linspace(0, 10, 100)
y = np.sin(x)

# show the plot
plt.show()
"""

# a more general example of animating a scatter plot in a function
def scatter_anim(x, y, func, interval=100, title='Scatter Plot Animation', xlabel='x', ylabel='y', save=False, filename='animation', fps=3, dpi=1000, **kwargs):
    '''
    Function to animate a scatter plot in 3D.
    Inputs:
        x: the x data
        y: the y data
        func: the function to plot
        interval: the interval between frames in milliseconds
        title: the title of the plot
        xlabel: the x axis label
        ylabel: the y axis label
        save: True or False to save the animation
        filename: the filename to save the animation as
        fps: the frames per second to save the animation as
        dpi: the dots per inch to save the animation as
        kwargs: additional keyword arguments to pass to the function
    '''
    # create a figure
    fig = plt.figure()

    # create a subplot
    ax = fig.add_subplot(1,1,1)

    # create a function that will do the plotting, where curr is the current frame
    def animate(curr):
        # clear the axes and redraw the plot for the next frame
        ax.clear()
        ax.scatter(x[:curr], y[:curr])
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        func(**kwargs)

    # create a variable that will contain the animation
    a = animation.FuncAnimation(fig, animate, frames=len(x), interval=interval)

    # save the animation if save is True
    if save:
        a.save(filename+'.gif', writer='pillow', fps=fps) # dpi=dpi
        
    # show the plot
    plt.show()
        
# create data
x = np.linspace(0, 1, 10)
y = np.sin(x)

# plot the animation
scatter_anim(x, y, plt.plot, title='Animation Example', xlabel='x', ylabel='y', save=True, filename='animation_example')