"""
Input: Phase-Field Fracture
===========================

"""

from phasefieldx.Materials.conversion import get_lambda_lame, get_mu_lame

class Input:
    """
    Class for managing phase-field fracture simulation parameters.

    This class encapsulates parameters related to phase-field fracture simulations
    and provides methods for setting and logging these parameters.

    Attributes:
        E (float): Young's modulus of the material.
        nu (float): Poisson's ratio of the material.
        Gc (float): Critical energy release rate.
        l (float): Length scale parameter.
        lambda_ (float): Lame's first parameter calculated from E and nu.
        mu (float): Shear modulus calculated from E and nu.
        degradation (str): Type of material degradation (isotropic or anisotropic).
        split_energy (str): Type of energy splitting (spectral, deviatoric, or none).
        degradation_function (str): Type of degradation function (e.g., quadratic).
        irreversibility (str): Indication of irreversibility of damage.
        fatigue (bool): Indicates if fatigue degradation is enabled.
        fatigue_degradation_function (str): Fatigue degradation function type.
        fatigue_val (float): Fatigue degradation value.
        k (int): A numerical parameter.
        min_stagger_iter (int): Minimum stagger iterations for numerical simulations.
        max_stagger_iter (int): Maximum stagger iterations for numerical simulations.
        stagger_error_tol (float): Tolerance for stagger error in simulations.
        save_solution_xdmf (bool): Indicates whether to save solutions in XDMF format.
        save_solution_vtu (bool): Indicates whether to save solutions in VTU format.
        result_folder_name (str): Name of the folder to save simulation results.

    Methods:
        __init__(...): Initialize the SimulationPhaseFieldFracture class with default parameters.
        save_log_info(logger): Logs the simulation parameters using the provided logger.
        __str__(): Returns a string representation of the simulation parameters.

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
                 min_stagger_iter=2,
                 max_stagger_iter=500,
                 stagger_error_tol=1e-8,
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
        self.min_stagger_iter = min_stagger_iter
        self.max_stagger_iter = max_stagger_iter
        self.stagger_error_tol = stagger_error_tol

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
        logger.info(
            f"  fatigue degradation function: {self.fatigue_degradation_function}")
        logger.info(f"  fatigue_val: {self.fatigue_val}")
        logger.info("Stagger settings: ")
        logger.info(f"  minimum stagger iterations: {self.min_stagger_iter }")
        logger.info(f"  maximum stagger iterations: {self.max_stagger_iter }")
        logger.info(f"  stagger error tolerance: {self.stagger_error_tol }")

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
