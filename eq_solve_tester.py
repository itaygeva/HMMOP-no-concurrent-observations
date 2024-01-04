import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
import math
from matplotlib.colors import Normalize
from matplotlib.patches import Circle



# Define the function representing the equation e^x = x
def func(Z, powers, value):
    f = np.exp(Z) - value * np.ones_like(Z)
    for power in powers:
        f -= np.power(Z, power)*(1/math.factorial(power))
    return np.abs(f)

def generate_min_image(value, powers, rect, res):
    # Generate a grid of complex numbers in a specified range
    real_range = np.linspace(-rect, rect, res)
    imag_range = np.linspace(-rect, rect, res)
    X, Y = np.meshgrid(real_range, imag_range)
    complex_plane = X + 1j * Y

    # Iterate through each complex number in the grid and find solutions
    f = func(complex_plane, powers, value)
    min_index_flat = np.argmin(f)
    min_index = np.unravel_index(min_index_flat, f.shape)
    norm = Normalize()


    # Plot the function
    plt.figure(figsize=(8, 6))
    plt.imshow(f, extent=(real_range.min(), real_range.max(), imag_range.min(), imag_range.max()), cmap='viridis',
               origin='lower', aspect='auto', norm=norm)
    plt.title(f'Function with Minimum Value with value {value}')

    # Add a dot at the minimum value location
    plt.scatter(X[min_index], Y[min_index], color='red', marker='o',
                label=f'Minimum: {f[min_index]:.2f}, at location: ({X[min_index]}, {Y[min_index]})')
    plt.legend()
    # Add a circle to the plot
    circle = Circle((0, 0), radius=1, edgecolor='red', facecolor='none')
    plt.gca().add_patch(circle)
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    cbar = plt.colorbar(label='Function Value')
    cbar_ticks = np.linspace(np.min(f), np.max(f), 5)
    cbar.set_ticks(cbar_ticks)
    cbar.set_ticklabels([f'{tick:.2f}' for tick in cbar_ticks])
    plt.show()
    cbar = None


powers = [0,1,2,3,4]
values = np.linspace(0, 0.01, 5)
for value in values:
    generate_min_image(value, powers, rect=1, res=200)
