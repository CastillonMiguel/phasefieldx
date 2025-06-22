"""
Input: Phase-Field Fracture
===========================

"""

from phasefieldx.Materials.conversion import get_lambda_lame, get_mu_lame
import csv

class Input:
    """
    Class for managing phase-field fracture simulation parameters.

    This class encapsulates parameters related to phase-field fracture simulations
    and provides methods for setting, logging, and exporting these parameters.

    Attributes
    ----------
    E : float
        Young's modulus of the material.
    nu : float
        Poisson's ratio of the material.
    Gc : float
        Critical energy release rate.
    l : float
        Length scale parameter.
    lambda_ : float
        Lame's first parameter calculated from E and nu.
    mu : float
        Shear modulus calculated from E and nu.
    degradation : str
        Type of material degradation (isotropic or anisotropic).
    split_energy : str
        Type of energy splitting (spectral, deviatoric, or none).
    degradation_function : str
        Type of degradation function (e.g., quadratic).
    irreversibility : str
        Indication of irreversibility of damage.
    fatigue : bool
        Indicates if fatigue degradation is enabled.
    fatigue_degradation_function : str
        Fatigue degradation function type.
    fatigue_val : float
        Fatigue degradation value.
    k : int
        A numerical parameter.
    save_solution_xdmf : bool
        Indicates whether to save solutions in XDMF format.
    save_solution_vtu : bool
        Indicates whether to save solutions in VTU format.
    results_folder_name : str
        Name of the folder to save simulation results.

    Methods
    -------
    __init__(...)
        Initialize the Input class with default parameters.
    save_log_info(logger)
        Log the simulation parameters using the provided logger.
    save_parameters_to_csv(filename="parameters.input")
        Save the simulation parameters to a two-column text file (tab-separated) for easy loading with pandas.
    __str__()
        Return a string representation of the simulation parameters.
    """

    def __init__(self,
                 E=210.0,
                 nu=0.3,
                 Gc=0.0027,
                 l=0.015,
                 degradation="isotropic",
                 split_energy="no",
                 degradation_function="quadratic",
                 irreversibility="no",
                 fatigue=False,
                 fatigue_degradation_function="asymptotic",
                 fatigue_val=0.05625,
                 k=0,
                 save_solution_xdmf=False,
                 save_solution_vtu=True,
                 results_folder_name="results"):
        """
        Initialize the SimulationPhaseFieldFracture class with default parameters.
        """
        self.E = E
        self.nu = nu
        self.Gc = Gc
        self.l = l
        self.lambda_ = get_lambda_lame(self.E, self.nu)
        self.mu = get_mu_lame(self.E, self.nu)
        self.degradation = degradation
        self.split_energy = split_energy
        self.degradation_function = degradation_function
        self.irreversibility = irreversibility
        self.fatigue = fatigue
        self.fatigue_degradation_function = fatigue_degradation_function
        self.fatigue_val = fatigue_val
        self.k = k

        self.save_solution_xdmf = save_solution_xdmf
        self.save_solution_vtu = save_solution_vtu
        self.results_folder_name = results_folder_name

    def save_log_info(self, logger):
        """
        Log the simulation parameters using the provided logger.

        Parameters:
            logger: An instance of a logging object.
        """
        logger.info("Material parameters:")
        logger.info(f"  E: {self.E}")
        logger.info(f"  nu: {self.nu}")
        logger.info(f"  Gc: {self.Gc}")
        logger.info(f"  l: {self.l}")
        logger.info(f"  k: {self.k}")
        logger.info(f"  lambda: {self.lambda_}")
        logger.info(f"  mu: {self.mu}")
        logger.info("Phase-field fracture model:")
        logger.info(f"  degradation: {self.degradation}")
        logger.info(f"  split_energy: {self.split_energy}")
        logger.info(f"  degradation_function: {self.degradation_function}")
        logger.info(f"  irreversibility: {self.irreversibility}")
        logger.info(f"  fatigue degradation function: {self.fatigue_degradation_function}")
        logger.info(f"  fatigue_val: {self.fatigue_val}")

    def save_parameters_to_csv(self, filename="parameters.input"):
        """
        Save the simulation parameters to a CSV file for easy loading with pandas.

        Parameters:
            filename (str): The name of the CSV file to save the parameters.
        """
        params = {
            "E": self.E,
            "nu": self.nu,
            "Gc": self.Gc,
            "l": self.l,
            "k": self.k,
            "lambda": self.lambda_,
            "mu": self.mu,
            "degradation": self.degradation,
            "split_energy": self.split_energy,
            "degradation_function": self.degradation_function,
            "irreversibility": self.irreversibility,
            "fatigue": self.fatigue,
            "fatigue_degradation_function": self.fatigue_degradation_function,
            "fatigue_val": self.fatigue_val,
            "save_solution_xdmf": self.save_solution_xdmf,
            "save_solution_vtu": self.save_solution_vtu,
            "results_folder_name": self.results_folder_name
        }
        with open(filename, "w") as f:
                for key, value in params.items():
                    f.write(f"{key}\t{value}\n")
     
    def __str__(self):
        """
        Return a string representation of the simulation parameters.

        Returns:
            str: A formatted string containing simulation parameter information.
        """
        parameter_info = [
            "Material parameters:",
            f"  E: {self.E}",
            f"  nu: {self.nu}",
            f"  Gc: {self.Gc}",
            f"  l: {self.l}",
            f"  k: {self.k}",
            f"  lambda: {self.lambda_}",
            f"  mu: {self.mu}",
            "Phase-field fracture model:",
            f"  degradation: {self.degradation}",
            f"  split_energy: {self.split_energy}",
            f"  degradation_function: {self.degradation_function}",
            f"  irreversibility: {self.irreversibility}",
            f"  fatigue degradation function: {self.fatigue_degradation_function}",
            f"  fatigue_val: {self.fatigue_val}"
        ]
        return "\n".join(parameter_info)
