"""
Input: Phase-Field
==================

"""


class Input:
    """
    Class for managing phase-field simulation parameters.

    This class encapsulates parameters related to phase-field fracture simulations
    and provides methods for setting, logging, and exporting these parameters.

    Attributes
    ----------
    l : float
        Length scale parameter.
    save_solution_xdmf : bool
        Indicates whether to save solutions in XDMF format.
    save_solution_vtu : bool
        Indicates whether to save solutions in VTU format.
    results_folder_name : str
        Name of the folder to save simulation results.

    Methods
    -------
    __init__(l=1.0, save_solution_xdmf=False, save_solution_vtu=True, results_folder_name="results")
        Initialize the Input class with default parameters.
    save_log_info(logger)
        Log the simulation parameters using the provided logger.
    save_parameters_to_csv(filename="parameters.input")
        Save the simulation parameters to a two-column text file (tab-separated) for easy loading with pandas.
    __str__()
        Return a string representation of the simulation parameters.
    """

    def __init__(self,
                 l=1.0,
                 save_solution_xdmf=False,
                 save_solution_vtu=True,
                 results_folder_name="results"):
        """
        Initialize the Input class with default parameters.
        """
        self.l = l

        self.save_solution_xdmf = save_solution_xdmf
        self.save_solution_vtu = save_solution_vtu
        self.results_folder_name = results_folder_name

    def save_log_info(self, logger):
        """
        Log the simulation parameters using the provided logger.

        Parameters:
            logger: An instance of a logging object.
        """
        logger.info("Parameters:")
        logger.info(f"  l: {self.l}")

    def save_parameters_to_csv(self, filename="parameters.input"):
        """
        Save the simulation parameters to a CSV file for easy loading with pandas.

        Parameters:
            filename (str): The name of the CSV file to save the parameters.
        """
        params = {
            "l": self.l,
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
            "Parameters:",
            f"  l: {self.l}",
        ]
        return "\n".join(parameter_info)
