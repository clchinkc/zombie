
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation

# create a bar chart animation

# x is a list of x-coordinates
# y is a list of lists of y-coordinates
# ticks is a list of x-axis labels
# x and y must have the same length

x = np.array([1, 2, 3, 4])
print(x)

y = [[1, 2, 3, 4], [1, 4, 9, 16], [1, 8, 27, 64]]
print(y)

ticks = ['one', 'two', 'three', 'four']
print(ticks)

# create a figure and axis
fig, ax = plt.subplots()

# set the title and labels
ax.set_title('Bar Chart Animation')
ax.set_xlabel('x')
ax.set_ylabel('y')

# create the bar chart
bars = ax.bar(x, y[0], tick_label=ticks)

# create timestep labels
text_box = ax.text(0.05, 0.9, '', transform=ax.transAxes)

# function to update the chart
def update(i):
    for j in range(len(bars)):
        bars[j].set_height(y[i][j])
    text_box.set_text('timestep: {}'.format(i))
        
# create the animation
anim = animation.FuncAnimation(fig, update, frames=len(y), interval=1000)

# save the animation as an gif file
anim.save('bar_chart_animation.gif', writer='pillow', fps=3)

# show the animation
plt.show()


# group everything into a function that accept x, y, and ticks as parameters
def bar_chart_animation(x, y, ticks):
    # create a figure and axis
    fig, ax = plt.subplots()

    # set the title and labels
    ax.set_title('Bar Chart Animation')
    ax.set_xlabel('x')
    ax.set_ylabel('y')

    # create the bar chart
    bars = ax.bar(x, y[0], tick_label=ticks)
    
    # create timestep labels
    text_box = ax.text(0.05, 0.9, '', transform=ax.transAxes)

    # function to update the chart
    def update(i):
        for j in range(len(bars)):
            bars[j].set_height(y[i][j])
            text_box.set_text('timestep: {}'.format(i))

    # create the animation
    anim = animation.FuncAnimation(fig, update, frames=len(y), interval=1000)

    # save the animation as an gif file
    anim.save('bar_chart_animation.gif', writer='pillow', fps=3)

    # show the animation
    # plt.show()

bar_chart_animation(x, y, ticks)






"""
class PopulationAnimator(Observer):
        
    def __init__(self, population: Population) -> None:
        self.subject = population
        self.subject.attach_observer(self)
        self.agent_history = []
        
    def update(self) -> None:
        self.agent_history.append(self.subject.agent_list)
        
    def display_observation(self, format='text'):
        if format == 'text':
            self.print_text_statistics()
        elif format == 'chart':
            self.chart_animation()
        
        # table
        
    def print_text_statistics(self):
        pass
    
    def population_animation(self):
        counts = []
        for i in range(len(self.agent_history)):
            cell_states = [individual.state for individual in self.agent_history[i]]
            counts.append([cell_states.count(state) for state in list(State)])
        print(np.array(State.value_list()))
        print(counts)
        print(State.name_list())
        self.bar_chart_animation(
        np.array(State.value_list()), 
                    counts, 
                    State.name_list())
    
    def bar_chart_animation(self, x, y, ticks):
        # create a figure and axis
        fig, ax = plt.subplots()

        # set the title and labels
        ax.set_title('Bar Chart Animation')
        ax.set_xlabel('x')
        ax.set_ylabel('y')

        # create the bar chart
        bars = ax.bar(x, y[0], tick_label=ticks)
        
        # create timestep labels
        text_box = ax.text(0.05, 0.9, '', transform=ax.transAxes)

        # function to update the chart
        def update(i):
            for j in range(len(bars)):
                bars[j].set_height(y[i][j])
            text_box.set_text(f'timestep = {i}')

        # create the animation
        anim = animation.FuncAnimation(fig, update, frames=len(y), interval=1000)

        # save the animation as an gif file
        anim.save('bar_chart_animation.gif', writer='pillow', fps=3)

        # show the animation
        plt.show()

    def chart_animation(self):
        fig, ax = plt.subplots()
        bar_chart = ax.bar(np.arange(len(State.value_list())), np.zeros(len(State.value_list())), tick_label=State.name_list())
        text_box = ax.text(0.02, 0.95, "", transform=ax.transAxes)
        def update(frame_number, bar_chart, text_box):
            cell_states = [individual.state for individual in self.agent_history[frame_number]]
            counts = {state: cell_states.count(state) for state in list(State)}
            for i, b in enumerate(bar_chart):
                b.set_height(counts[State(i)])
            text_box.set_text(f"Step = {frame_number}")
        ani = animation.FuncAnimation(fig, func=update, frames=len(self.agent_history), fargs=(bar_chart, text_box), interval=100)
        ani.save('population_animation.gif', writer='imagemagick')
        plt.show()

    
    def chart_animate(self):
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        def animate(i):
            cell_states = [individual.state for individual in self.agent_history[i]]
            counts = {state: cell_states.count(state) for state in list(State)}
            print(np.array(State.value_list()))
            print(list(counts.values()))
            print(State.name_list())
            plt.bar(np.asarray(State.value_list()), 
                    list(counts.values()), 
                    tick_label=State.name_list())
            plt.text(0.02, 0.95, f"Step = {i}", transform=ax.transAxes)
            return ax
        ani = animation.FuncAnimation(fig, animate, interval=100)
        # Save the animation
        ani.save('population_animation.gif', fps=10)
        # Show the animation
        plt.show()
    
    def animate_chart(self):
        # Create a figure
        fig, ax = plt.subplots()
        # Define the update function
        def update(frame_number):
            # Plot the bar chart
            cell_states = [individual.state for individual in self.agent_history[frame_number]]
            counts = {state: cell_states.count(state) for state in list(State)}
            ax.clear()
            ax.bar(np.asarray(State.value_list()), list(
                counts.values()), tick_label=State.name_list())
            # Plot the text box
            ax.text(0.02, 0.95, f"Step = {frame_number}", transform=ax.transAxes)
            return ax
        # Create the animation
        ani = animation.FuncAnimation(fig, update, frames=len(self.agent_history), interval=10, repeat=False)
        # Save the animation
        ani.save('population_animation.gif', fps=10)
        # Show the animation
        plt.show()
        # Clear the current figure
        plt.cla()
        
    # animation
    def print_chart_animation(self):
        # Create a figure
        fig, ax = plt.subplots()
        # Create a bar chart
        bar_chart = ax.bar(np.arange(len(State.value_list())), np.zeros(len(State.value_list())), tick_label=State.name_list())
        # Create a text box to display the number of steps
        text_box = ax.text(0.02, 0.95, "", transform=ax.transAxes)
        # Define the update function
        def update(frame_number, bar_chart, text_box):
            # Update the bar chart
            cell_states = [individual.state for individual in self.agent_history[frame_number]]
            counts = {state: cell_states.count(state) for state in list(State)}
            plt.bar(np.array(State.value_list()), list(
                    counts.values()), tick_label=State.name_list())
            # Update the text box
            text_box.set_text(f"Step = {frame_number}")
        # Create an animation
        ani = animation.FuncAnimation(fig, func=update, frames=len(self.agent_history), fargs=(bar_chart, text_box), interval=1)
        # Save the animation
        ani.save('population_animation.gif', writer='imagemagick')
        # Clear the current figure
        plt.cla()
            
    # animation
    def print_school_animation(self):
        # Create a figure
        fig = plt.figure()
        # Create a subplot
        ax = fig.add_subplot(1, 1, 1)
        # Create a scatter plot
        sc = ax.scatter([], [], s=10)
        # Create a text label
        label = ax.text(0.02, 0.95, "", transform=ax.transAxes)
        # Create an animation
        ani = animation.FuncAnimation(fig, self.animate, frames=len(
            self.agent_history), interval=100, blit=True, repeat=False, fargs=(sc, label))
        # Show the plot
        plt.show()
        
    def animate(self, i, sc, label):
        # Get the current state of the population
        agent_list = self.agent_history[i]
        # Update the scatter plot
        sc.set_offsets(np.asarray([agent.position for agent in agent_list]))
        sc.set_color([agent.color for agent in agent_list])
        # Update the text label
        label.set_text(f"Day {i}")
        return sc, label
    
    def animation_scatter(self):
        # Create a figure
        fig = plt.figure()
        # Create a subplot
        ax = fig.add_subplot(1, 1, 1)
        # Create a scatter plot
        sc = ax.scatter([], [], s=100)
        # Create a text label
        text = ax.text(0.02, 0.95, "", transform=ax.transAxes)
        # Create a function to update the scatter plot
        def animate(i):
            # Get the current state of the population
            agent_list = self.agent_history[i]
            # Update the scatter plot
            sc.set_offsets(np.asarray([agent.location for agent in agent_list]))
            # Update the text
            text.set_text(f"Day {i}")
            # Return the artists set
            return sc, text
        # Create an animation
        anim = animation.FuncAnimation(fig, animate, frames=len(self.agent_history), interval=100)
        # Show the plot
        plt.show()

"""