import numpy as np
import matplotlib.pyplot as plt



def phi(x,l,a):
    one_div_exp2adivl_one = 1/(np.exp(2*a/l)+1)
    return np.exp(-abs(x)/l) + one_div_exp2adivl_one * 2* np.sinh(np.abs(x)/l)

def gradphi(x,l,a):
    one_div_exp2adivl_one = 1/(np.exp(2*a/l)+1)
    return -np.sign(x)/l * np.exp(-abs(x)/l) + one_div_exp2adivl_one * np.sign(x)/l * 2* np.cosh(np.abs(x)/l)


a = 1.0
l = 0.1

x = np.linspace(-a,a,10000)


# PHI ####################################################
fig, ax_phi = plt.subplots() 
ax_phi.plot(x, phi(x, l, a) ,'k-',  label = '$l=0.1$')
ax_phi.plot(x, phi(x, 0.25, a) ,'b-',  label = '$l=0.5$')

ax_phi.set_xlim(-a*1.05,a*1.05) 
ax_phi.set_ylim(-0.05,1*1.05) 
ax_phi.set_xlabel('x' )  
ax_phi.set_ylabel('$\phi(x)$') 
ax_phi.grid(color='k', linestyle='-', linewidth=0.3)  
ax_phi.legend() 

# Save PHI graph in a specific folder
folder_path_images = '../images/'
file_name_phi = 'phi_graph.png'
fig.savefig(folder_path_images + file_name_phi, dpi=300)  # Adjust dpi (dots per inch) as needed


# GRADPHI ###################################################
fig, ax_gradphi = plt.subplots() 
ax_gradphi.plot(x, gradphi(x, l, a) ,'k--',  label = '$l=0.1$')
ax_gradphi.plot(x, gradphi(x, 0.25, a) ,'b--',  label = '$l=0.5$')

#ax_gradphi.set_xlim(-a*1.05,a*1.05) 
#ax_gradphi.set_ylim(-0.05,1*1.05) 
ax_gradphi.set_xlabel('x' )  
ax_gradphi.set_ylabel('$\phi´(x)$') 
ax_gradphi.grid(color='k', linestyle='-', linewidth=0.3)  
ax_gradphi.legend() 
fig.savefig(folder_path_images + 'gradphi_graph.png', dpi=300)  # Adjust dpi (dots per inch) as needed

plt.show()