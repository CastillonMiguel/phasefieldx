"""
ReferenceResult
===============

This module provides classes for handling and post-processing PhaseFieldX simulation results.

"""

import pandas as pd
import os

from phasefieldx.files import get_filenames_in_folder, get_foldernames_in_folder
from phasefieldx.PostProcessing.paraview import ParaviewResult


class AllResults:
    """
    Class to handle PhaseFieldX results from a given folder.
  
    This class automatically loads and organizes various types of result files
    including convergence data, energy files, reaction forces, degrees of freedom,
    input parameters, and ParaView visualization files from a PhaseFieldX simulation
    output directory.

    Parameters
    ----------
    folder_path : str
        Path to the folder containing PhaseFieldX results. The folder should contain
        various output files with extensions like .conv, .energy, .reaction, .dof, 
        .input, and potentially ParaView solution folders.

    Attributes
    ----------
    folder_path : str
        Path to the results folder.
    input : dict
        Dictionary containing input parameters from .input files.
    convergence_files : dict
        Dictionary of pandas DataFrames containing convergence data from .conv files.
    energy_files : dict
        Dictionary of pandas DataFrames containing energy data from .energy files.
    reaction_files : dict
        Dictionary of pandas DataFrames containing reaction force data from .reaction files.
    dof_files : dict
        Dictionary of pandas DataFrames containing degree of freedom data from .dof files.
    file_names : list
        List of all file names found in the results folder.
    folder_names : list
        List of all subdirectory names found in the results folder.
    paraview_filenames : list, optional
        List of ParaView solution file names, if available.
    label : str
        Label for plotting and identification purposes. Default is 'label'.
    color : str
        Color specification for plotting. Default is 'k' (black).
    paraview : ParaviewResult, optional
        ParaviewResult object for handling visualization data.

    Methods
    -------
    get_paraview(path)
        Initialize ParaView result handling for the specified path.
    set_label(label)
        Set the label for this result set.
    set_color(color)
        Set the color for plotting this result set.

    Examples
    --------
    >>> # Load results from a simulation folder
    >>> results = AllResults("./simulation_results")
    >>> 
    >>> # Set label and color for plotting
    >>> results.set_label("My Simulation")
    >>> results.set_color("red")
    >>> 
    >>> # Access energy data
    >>> energy_data = results.energy_files['total.energy']
    >>> print(energy_data.head())
    >>> 
    >>> # Access input parameters
    >>> parameters = results.input['parameters.input']
    >>> print(f"Length scale: {parameters.get('l', 'N/A')}")

    Notes
    -----
    The class automatically categorizes files based on their extensions:
    
    - `.conv` files: Convergence information
    - `.energy` files: Energy-related data
    - `.reaction` files: Reaction force data
    - `.dof` files: Degrees of freedom data
    - `.input` files: Input parameters (converted to dictionary format)
    
    ParaView visualization files are detected if a "paraview-solutions_vtu" 
    subdirectory exists in the results folder.
    """

    def __init__(self, folder_path):
        """
        Initialize the AllResults class.

        Parameters
        ----------
        folder_path : str
            Path to the folder containing PhaseFieldX results. The folder should
            exist and contain simulation output files.

        Raises
        ------
        FileNotFoundError
            If the specified folder_path does not exist.
        PermissionError
            If the folder cannot be accessed due to permission restrictions.
        """
        self.folder_path = folder_path
        
        # Initialize data containers
        self.input = {}
        self.convergence_files = {}
        self.energy_files = {}
        self.reaction_files = {}
        self.dof_files = {}

        # Get all files and folders in the results directory
        self.file_names = get_filenames_in_folder(self.folder_path)

        # Default plotting attributes
        self.label = 'label'
        self.color = 'k'

        # Process each file based on its extension
        for file_name in self.file_names:

            file_path = os.path.join(self.folder_path, file_name)

            if file_name.endswith(".conv"):
                self.convergence_files[file_name] = pd.read_csv(
                    file_path, sep='\t')
            elif file_name.endswith(".energy"):
                self.energy_files[file_name] = pd.read_csv(file_path, sep='\t')
            elif file_name.endswith(".reaction"):
                self.reaction_files[file_name] = pd.read_csv(
                    file_path, sep='\t')
            elif file_name.endswith(".dof"):
                self.dof_files[file_name] = pd.read_csv(file_path, sep='\t')
            elif file_name.endswith(".input"):
                df = pd.read_csv(file_path, sep='\t', header=None, comment='/')
                # Convert to dictionary format for easy parameter access
                self.input[file_name] = dict(zip(df[0], df[1]))

        # Process subdirectories
        self.folder_names = get_foldernames_in_folder(self.folder_path)

        # Check for ParaView visualization files
        try:
            if self.folder_names[0] == "paraview-solutions_vtu":
                self.paraview_filenames = get_filenames_in_folder(
                    os.path.join(self.folder_path, "paraview-solutions_vtu"))
                # self.paraview = ParaviewResult(self.folder_path, np.array([len(paraview_filenames )-1]))

        except Exception:
            print("No paraview solution files found")

    def get_paraview(self, path):
        """
        Initialize ParaView result handling for visualization.

        Parameters
        ----------
        path : str
            Path to the ParaView solution files or results folder containing
            ParaView visualization data.

        Notes
        -----
        This method creates a ParaviewResult object that can be used for
        post-processing and visualization of the simulation results.
        """
        self.paraview = ParaviewResult(path)

    def set_label(self, label):
            """
            Set the label for this result set.

            Parameters
            ----------
            label : str
                Label to identify this result set in plots and analysis.
                This label will be used in plot legends and other visual representations.

            Examples
            --------
            >>> results = AllResults("./simulation_results")
            >>> results.set_label("High Resolution Mesh")
            """
            self.label = label

    def set_color(self, color):
        """
        Set the color for plotting this result set.

        Parameters
        ----------
        color : str
            Color specification for matplotlib plotting. Can be a single character
            ('r', 'g', 'b', 'k', etc.), color name ('red', 'blue', etc.), or
            hex color code ('#FF0000', etc.).

        Examples
        --------
        >>> results = AllResults("./simulation_results")
        >>> results.set_color("red")
        >>> # or
        >>> results.set_color("#FF0000")
        >>> # or
        >>> results.set_color("r")
        """
        self.color = color

    def __str__(self):
        """
        Return a string representation of the AllResults object.

        Returns
        -------
        str
            A formatted string containing information about all loaded files
            organized by category (energy files, reaction files, DOF files).
            This provides a quick overview of what data is available in the
            result set.

        Examples
        --------
        >>> results = AllResults("./simulation_results")
        >>> print(results)
        
         # Files #####################
        
         # Energy Files:--------------
        total.energy
        
         # Reaction Files:-----------
        left.reaction
        right.reaction
        
         # DOF Files:----------------
        top.dof
        """
        result_str = "\n\n # Files #####################\n"
        result_str += "\n\n # Energy Files:--------------\n"
        result_str += "\n".join(self.energy_files.keys())
        result_str += "\n\n # Reaction Files:-----------\n"
        result_str += "\n".join(self.reaction_files.keys())
        result_str += "\n\n # DOF Files:----------------\n"
        result_str += "\n".join(self.dof_files.keys())

        return result_str
