import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
import matplotlib.colors as mcolors


# Generate x values
x = np.linspace(-1, 1, 10000)

# Plotting
social=True
if social:
    plt.figure(figsize=(12.8, 6.4))  # Set figure size in inches for 1280x640 pixels
else:
    plt.figure(figsize=(10, 10))
exp_data = np.exp(-abs(x/0.225))
plt.plot( exp_data,  x, color='white', linewidth=5.0)
plt.plot(-exp_data, -x, color='white', linewidth=5.0)

# Add a circle in the backgroun
circle = plt.Circle((0, 0), 1, color='darkred', fill=True, linestyle='-', linewidth=5.0)
plt.gca().add_artist(circle)
circle = plt.Circle((0, 0), 1, color='black', fill=False, linestyle='-', linewidth=5.0)
plt.gca().add_artist(circle)

x2 = np.linspace(-1.1*np.sqrt(2*0.5**2), np.sqrt(2*0.5**2), 10000)
y_tanh_grad = 0.5*np.tanh(x / 0.1)

plt.plot(x, y_tanh_grad, color='white', linewidth=10.0)
plt.plot(x, y_tanh_grad, color='black', linewidth=20.0)
plt.plot(x, y_tanh_grad, color='white', linewidth=10.0)


if social==False:
    plt.text(1.1, 0, 'PhaseFieldX', color='black', fontsize=24*3.5, va='center', ha='left')
#plt.subplots_adjust(right=0.3, top=1.0)


# Set aspect ratio to be equal
plt.gca().set_aspect('equal', adjustable='box')

# Remove all axes
plt.axis('off')

# Save or show the plot
plt.tight_layout()
#plt.savefig('logo.png', transparent=True)  # Save the plot as an image file

if social:
    plt.savefig('logo_social.png', transparent=True)  # Save the plot as an image file
else:
    plt.savefig('logo.png', transparent=True)  # Save the plot as an image file
plt.show()
