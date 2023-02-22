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


def print_school_animation(x, y):
    # Create a figure
    fig, ax = plt.subplots(1, 1)
    # Create an animation function
    def animate(i, sc, label):
        # Update the scatter plot
        sc.set_offsets(np.c_[x[:i], y[:i]])
        # Set the label
        label.set_text("t = {}".format(i))
        # Return the artists set
        return sc, label
    # Create a scatter plot
    sc = ax.scatter(x, y)
    # Create a label
    label = ax.text(0.05, 0.9, "", transform=ax.transAxes)
    # Create the animation object
    anim = animation.FuncAnimation(fig, animate, frames=len(x), interval=100, blit=True, repeat=False, fargs=(sc, label))
    # Save the animation
    anim.save('animation.gif', writer='pillow', fps=3, dpi=10)
    
    # Show the plot
    plt.show()
    
# print_school_animation(x, y)


import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

# Fixing random state for reproducibility
np.random.seed(19680801)


def random_walk(num_steps, max_step=0.05):
    """Return a 3D random walk as (num_steps, 3) array."""
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
