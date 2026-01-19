"""
Input: Allen-Cahn
=================

"""


class Input:
    """
    Class for managing phase-field simulation parameters.

    This class encapsulates parameters related to phase-field fracture simulations
    and provides methods for setting and logging these parameters.

    Attributes:
        Gc (float): Critical energy release rate.
        l (float): Length scale parameter.
        label (str): A label associated with the simulation.
        color (str): A color associated with the simulation.

    Methods:
        __init__(): Initialize the SimulationPhaseFieldFracture class with default parameters.
        save_log_info(logger): Logs the simulation parameters using the provided logger.
        __str__(): Returns a string representation of the simulation parameters.

    """

    def __init__(self,
                 l=1.0,
                 mobility=1.0,
                 save_solution_xdmf=False,
                 save_solution_vtu=True,
                 result_folder_name="results"):
        """
        Initialize the SimulationPhaseFieldFracture class with default parameters.
        """
        self.l = l
        self.mobility = mobility
        self.save_solution_xdmf = save_solution_xdmf
        self.save_solution_vtu = save_solution_vtu
        self.results_folder_name = result_folder_name

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
