# tutorial for animating a scatter plot
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np


# define a function to animate a matplotlib plot
def scatter_anim(x, y, interval=100, title='Scatter Plot Animation', xlabel='x', ylabel='y', save=False, filename='animation', fps=3, dpi=10, **kwargs):
    '''
    Function to animate a scatter plot in 3D.
    Inputs:
        x: the x data
        y: the y data
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
# scatter_anim(x, y, title='Animation Example', xlabel='x', ylabel='y', save=True, filename='animation_example')

# a general function to animate a matplotlib plot of any type, e.g. line, scatter, etc.
def animation_function(x, y, func, mode, interval=100, title='Plot Animation', xlabel='x', ylabel='y', save=False, filename='animation', fps=3, dpi=10, **kwargs):
    '''
    Function to animate a plot in 3D.
    Inputs:
        x: the x data
        y: the y data
        func: the function to plot the data with
        mode: the mode to plot the data with, either 'step' or 'progress'
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
        if mode == "step":
            func(x[curr], y[curr], **kwargs)
        elif mode == "progress":
            func(x[:curr], y[:curr], **kwargs)
        else:
            raise ValueError("mode must be either 'step' or 'progress'")
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_xlim(0, max(x))
        ax.set_ylim(min(y), max(y))
        
    # create a variable that will contain the animation
    anim = animation.FuncAnimation(fig, animate, frames=len(x), interval=interval, repeat=False)
    
    # save the animation if save is True
    if save:
        anim.save(filename+'.gif', writer='pillow', fps=fps, dpi=dpi)
        
    # show the plot
    plt.show()
    
# plot the animation
# scatter plot
# anim(x, y, plt.scatter, mode="step", title='Animation Example', xlabel='x', ylabel='y', save=True, filename='scatter_animation')
# line plot
animation_function(x, y, plt.plot, mode="progress", title='Animation Example', xlabel='x', ylabel='y', save=True, filename='line_animation')
# bar plot
# anim(x, y, plt.bar, mode="progress", title='Animation Example', xlabel='x', ylabel='y', save=True, filename='bar_animation')