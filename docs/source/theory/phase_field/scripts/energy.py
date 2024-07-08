import numpy as np
import matplotlib.pyplot as plt



def phi(x,l,a):
    one_div_exp2adivl_one = 1/(np.exp(2*a/l)+1)
    return np.exp(-abs(x)/l) + one_div_exp2adivl_one * 2* np.sinh(np.abs(x)/l)

def gradphi(x,l,a):
    one_div_exp2adivl_one = 1/(np.exp(2*a/l)+1)
    return -np.sign(x)/l * np.exp(-abs(x)/l) + one_div_exp2adivl_one * np.sign(x)/l * 2* np.cosh(np.abs(x)/l)


def W_phi(l,a):
    a_div_l = a/l
    tanh_a_div_l = np.tanh(a_div_l)
    return 0.5*tanh_a_div_l + 0.5*a_div_l*(1.0-tanh_a_div_l**2)

def W_gradphi(l,a):
    a_div_l = a/l
    tanh_a_div_l = np.tanh(a_div_l)
    return 0.5*tanh_a_div_l - 0.5*a_div_l*(1.0-tanh_a_div_l**2)

def W(l,a):
    a_div_l = a/l
    tanh_a_div_l = np.tanh(a_div_l)
    return tanh_a_div_l


gc = 1.0

a = 5.0
l = np.linspace(0.1,2*a,100)




# PHI ####################################################
fig, energy = plt.subplots() 
energy.plot(l/a, W_phi(l,a),     'r-',  label = '$W_{\phi}$')
energy.plot(l/a, W_gradphi(l,a) ,'b-',  label = '$W_{V \phi}$')
energy.plot(l/a, W(l,a) ,'k-',  label = '$W$')
energy.set_xlabel('l/a' )  
energy.set_ylabel('energy') 
energy.grid(color='k', linestyle='-', linewidth=0.3)  
energy.legend() 

# Save PHI graph in a specific folder
folder_path_images = '../images/'
file_name_phi = 'energy_l_vs_energy.png'
fig.savefig(folder_path_images + file_name_phi, dpi=300)  # Adjust dpi (dots per inch) as needed


# PHI ####################################################
fig, energy2 = plt.subplots() 
energy2.plot(a/l, W_phi(l,a),     'r-',  label = '$W_{\phi}$')
energy2.plot(a/l, W_gradphi(l,a) ,'b-',  label = '$W_{V \phi}$')
energy2.plot(a/l, W(l,a) ,'k-',  label = '$W$')
energy2.set_xlabel('a/l' )  
energy2.set_ylabel('energy') 
energy2.grid(color='k', linestyle='-', linewidth=0.3)  
energy2.legend() 

# Save PHI graph in a specific folder
folder_path_images = '../images/'
file_name_phi = 'energy_al_vs_energy.eps'
fig.savefig(folder_path_images + file_name_phi, dpi=300)  # Adjust dpi (dots per inch) as needed



plt.show()