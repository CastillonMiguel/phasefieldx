"""
Input: Elasticity
=================

"""

from phasefieldx.Materials.conversion import get_lambda_lame, get_mu_lame


class Input:
    """
    Class for managing elasticity simulation parameters.

    This class encapsulates parameters related to elasticity simulations
    and provides methods for setting, logging, and exporting these parameters.

    Attributes
    ----------
    E : float
        Young's modulus of the material.
    nu : float
        Poisson's ratio of the material.
    lambda_ : float
        Lame's first parameter calculated from E and nu.
    mu : float
        Shear modulus calculated from E and nu.
    save_solution_xdmf : bool
        Indicates whether to save solutions in XDMF format.
    save_solution_vtu : bool
        Indicates whether to save solutions in VTU format.
    results_folder_name : str
        Name of the folder to save simulation results.

    Methods
    -------
    __init__(E=210.0, nu=0.3, save_solution_xdmf=False, save_solution_vtu=True, results_folder_name="results")
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
                 save_solution_xdmf=False,
                 save_solution_vtu=True,
                 results_folder_name="results"):
        """
        Initialize the SimulationPhaseFieldFracture class with default parameters.
        """
        self.E = E
        self.nu = nu
        self.lambda_ = get_lambda_lame(self.E, self.nu)
        self.mu = get_mu_lame(self.E, self.nu)

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
        logger.info(f"  lambda: {self.lambda_}")
        logger.info(f"  mu: {self.mu}")

    def save_parameters_to_csv(self, filename="parameters.input"):
        """
        Save the simulation parameters to a CSV file for easy loading with pandas.

        Parameters:
            filename (str): The name of the CSV file to save the parameters.
        """
        params = {
            "E": self.E,
            "nu": self.nu,
            "lambda": self.lambda_,
            "mu": self.mu,
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
            f"  lambda: {self.lambda_}",
            f"  mu: {self.mu}"
        ]
        return "\n".join(parameter_info)
