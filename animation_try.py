# tutorial for animating a scatter plot
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np


# define a function to animate a scatter plot
def scatter_anim(x, y, func, interval=100, title='Scatter Plot Animation', xlabel='x', ylabel='y', save=False, filename='animation', fps=3, dpi=10, **kwargs):
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
    # create a figure and axes
    fig, ax = plt.subplots()
    
    # create a function that will do the plotting, where curr is the current frame
    def animate(curr):
        # clear the axes and redraw the plot for the next frame
        ax.clear()
        ax.scatter(x[:curr], y[:curr])
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_xlim(0, max(x))
        ax.set_ylim(min(y), max(y))
        func(**kwargs)

    # create a variable that will contain the animation
    anim = animation.FuncAnimation(fig, animate, frames=len(x), interval=interval, repeat=False)

    # save the animation if save is True
    if save:
        anim.save(filename+'.gif', writer='pillow', fps=fps, dpi=dpi)
        
    # show the plot
    plt.show()
        
# create data
x = np.linspace(0, 10, 100)
y = np.sin(x)

# plot the animation
# scatter_anim(x, y, plt.plot, title='Animation Example', xlabel='x', ylabel='y', save=True, filename='animation_example')
