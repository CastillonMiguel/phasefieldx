import matplotlib.pyplot as plt
import numpy as np
# Data for the bar chart
a = 1  # Set the value of 'a'
b = 2  # Set the value of 'b'



# Create a figure and axis
fig, ax = plt.subplots(frameon=False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.set_xticks([])
ax.set_yticks([])


# Add a horizontal line at the center
ax.plot([-a, b], [0, 0], color='black', linewidth=3, linestyle='-')
ax.plot([-a, b], [0, 0], color='black', linewidth=3, marker='s')
ax.plot( 0, 0, color='black', linewidth=3, marker='o')
ax.plot([0, 0], [-0.05, 0.25], color='black', linewidth=1, linestyle='--')

l=0.1
x = np.arange(-a, b,0.01)
phi = 0.2*np.exp(-abs(x)/l)+0.025
ax.plot(x, phi, color='red', linewidth=2, linestyle='--')

# Add text annotations at (-a, 0) and (a, 0)
#ax.text(-a, 0.05, '-a', color='black', ha='center')
#ax.text(b, 0.05, 'b', color='black', ha='center')

# Add text annotations at (-a, 0) and (a, 0)
h=0.025

ax.text(-a*0.5, -h*1.5, '$a$', color='black', ha='center')
ax.text(b*0.5, -h*1.5, '$b$', color='black', ha='center')

ax.annotate("", xy=(-a, -h), xytext=(0.0, -h),
            arrowprops=dict(arrowstyle="<->", linewidth=2))

ax.annotate("", xy=(0.0, -h), xytext=(b, -h),
            arrowprops=dict(arrowstyle="<->", linewidth=2))

ax.annotate("", xy=(-a, -2.5*h), xytext=(b, -2.5*h),
            arrowprops=dict(arrowstyle="<->", linewidth=2))
ax.text(0.0, -h*3.25, '$L$', color='black', ha='center')


ax.set_xlim(-1.1*a, 1.1 * b)
ax.set_ylim(-0.1, 0.3)


# Show the plot
plt.show()

folder_path_images = '../images/'
file_name_phi = 'bar_graph_ab.png'
fig.savefig(folder_path_images + file_name_phi, dpi=300)  # Adjust dpi (dots per inch) as needed
