"""
ReferenceResult
===============

AllResults: Class to handle PhaseFieldX results from a given folder.

"""

import pandas as pd
import os

from phasefieldx.files import get_filenames_in_folder, get_foldernames_in_folder
from phasefieldx.PostProcessing.paraview import ParaviewResult


class AllResults:
    """
    Class to handle PhaseFieldX results from a given folder.
    """

    def __init__(self, folder_path):
        """
        Initialize the AllResultsIris class.

        Parameters:
            folder_path (str): Path to the folder containing PhaseFieldX results.

        """
        self.folder_path = folder_path

        self.input = {}
        self.convergence_files = {}
        self.energy_files = {}
        self.reaction_files = {}
        self.dof_files = {}

        self.file_names = get_filenames_in_folder(self.folder_path)

        self.label = 'label'
        self.color = 'k'

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
                # Convert to dictionary
                self.input[file_name] = dict(zip(df[0], df[1]))

        self.folder_names = get_foldernames_in_folder(self.folder_path)

        try:
            if self.folder_names[0] == "paraview-solutions_vtu":
                self.paraview_filenames = get_filenames_in_folder(
                    os.path.join(self.folder_path, "paraview-solutions_vtu"))
                # self.paraview = ParaviewResult(self.folder_path, np.array([len(paraview_filenames )-1]))

        except Exception:
            print("No paraview solution")

    def get_paraview(self, path):
        self.paraview = ParaviewResult(path)

    def set_label(self, label):
        self.label = label

    def set_color(self, color):
        self.color = color

    def __str__(self):
        """
        Return a string representation of the AllResultsIris object.

        Returns:
            str: String representation of the AllResultsIris object.

        """
        result_str = "\n\n # Files #####################\n"
        result_str += "\n\n # Energy Files:--------------\n"
        result_str += "\n".join(self.energy_files)
        result_str += "\n\n # Reaction Files:-----------\n"
        result_str += "\n".join(self.reaction_files)
        result_str += "\n\n # DOF Files:----------------\n"
        result_str += "\n".join(self.dof_files)

        return result_str
