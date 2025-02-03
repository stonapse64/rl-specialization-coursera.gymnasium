'''Figure 2.1: An example bandit problem from the 10-armed testbed. The true value q⇤(a) of
each of the ten actions was selected according to a normal distribution with mean zero and unit
variance, and then the actual rewards were selected according to a mean q⇤(a), unit-variance
normal distribution, as suggested by these gray distributions.'''

import matplotlib.pyplot as plt
import numpy as np

# Generate the data
arms = 10
arms_q_values = np.random.normal(0, 1, arms)
timesteps = 10000
arms_values_over_time = np.zeros(timesteps)

arm = np.min(arms_q_values)
for i in range(timesteps):
    arms_values_over_time[i] = np.random.normal(arm, 1)

# Create a violin plot based on the data in arms_values_over_time
plt.violinplot(arms_values_over_time)
plt.title('Violin Plot of Arms Values Over Time')
plt.xlabel('Arms')
plt.ylabel('Values')

# Calculate the mean of the distribution
mean_value = np.mean(arms_values_over_time)

# Add a shorter marker/error line for the mean of the distribution
plt.axhline(mean_value, color='r', linestyle='--', linewidth=1, xmin=0.45, xmax=0.55)

# Get the current axis
ax = plt.gca()

# Get the right end of the error line
xmax = ax.get_xlim()[1] * 0.55
print (xmax, ax.get_xlim()[1])

# Add text relative to the right end of the error line
plt.text(1.45 * xmax + 0.02 * ax.get_xlim()[1], mean_value, f'q({np.argmin(arms_q_values)})', color='r', verticalalignment='center')

# Show the plot
plt.show()