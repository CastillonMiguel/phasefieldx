"""
File Utilities
==============

This module provides functions for working with files and folders, including
file search, text replacement, and file and folder deletion.

"""

import os
from typing import List
import shutil


def get_filenames_in_folder(folder_path: str) -> List[str]:
    """
    Get the filenames in the specified folder.

    Parameters
    ----------
    folder_path : str
        The path of the folder.

    Returns
    -------
    List[str]
        A list of filenames in the folder.

    Examples
    --------
    >>> folder_path = 'path/to/your/folder'
    >>> filenames = get_filenames_in_folder(folder_path)
    >>> for filename in filenames:
    ...     print(filename)
    """
    filenames = []
    for file in os.listdir(folder_path):
        if os.path.isfile(os.path.join(folder_path, file)):
            filenames.append(file)
    return filenames


def get_foldernames_in_folder(folder_path: str) -> List[str]:
    """
    Get the folders in the specified folder.

    Parameters
    ----------
    folder_path : str
        The path of the folder.

    Returns
    -------
    List[str]
        A list of folders in the folder.

    Examples
    --------
    >>> folder_path = 'path/to/your/folder'
    >>> folders = get_foldernames_in_folder(folder_path)
    >>> for folder_name in folders:
    ...     print(folder_name)
    """
    folders = []
    for item in os.listdir(folder_path):
        if os.path.isdir(os.path.join(folder_path, item)):
            folders.append(item)
    return folders


def find_files_with_extension(folder_path: str, extension: str) -> List[str]:
    """
    Find files with the specified extension in the given folder.

    Parameters
    ----------
    folder_path : str
        The path of the folder.
    extension : str
        The file extension to search for.

    Returns
    -------
    List[str]
        A list of file paths with the matching extension.

    Examples
    --------
    >>> folder_path = 'path/to/your/folder'
    >>> extension = '.txt'
    >>> files_with_extension = find_files_with_extension(folder_path,extension)
    >>> for file_path in files_with_extension:
    ...     print(file_path)
    """
    matching_files = []
    matching_path_files = []
    for files in os.walk(folder_path):
        for file in files:
            if file.endswith(extension):
                matching_files.append(os.path.join(file))
                matching_path_files.append(os.path.join(folder_path, file))
    return sorted(matching_path_files)


def read_last_line(file_path):
    """
    Read the last line of a specified file.

    Parameters
    ----------
    file_path : str
        The path to the file to be read.

    Returns
    -------
    str or None
        The last line of the file as a string, stripped of leading and trailing
        whitespace.
        Returns None if the file is empty.

    Examples
    --------
    >>> file_path = 'path/to/your/file.txt'
    >>> last_line = read_last_line(file_path)
    >>> if last_line is not None:
    ...     print("Last line:", last_line)
    ... else:
    ...     print("The file is empty.")
    """
    with open(file_path, 'r') as file:
        lines = file.readlines()
        if lines:
            return lines[-1].strip()
        else:
            return None


def read_last_lines(file_path, num_lines):
    """
    Read the specified number of lines from the end of a file.

    Parameters
    ----------
    file_path : str
        The path to the file to be read.
    num_lines : int
        The number of lines to read from the end of the file.

    Returns
    -------
    list
        A list of strings representing the last `num_lines` lines of the file.
        If the file has fewer lines than `num_lines`, it will return all
        available lines.

    Examples
    --------
    >>> file_path = 'path/to/your/file.txt'
    >>> num_lines = 5
    >>> last_lines = read_last_lines(file_path, num_lines)
    >>> for line in last_lines:
    ...     print(line)
    """
    with open(file_path, 'r') as file:
        lines = file.readlines()
        last_lines = lines[-num_lines:]
        return [line.strip() for line in last_lines]


def replace_text(original, target, list_replace):
    """
    Replaces specified patterns with corresponding strings in a file.

    Parameters
    ----------
    original : str
        Path to the original file.
    target : str
        Path to the target file where the modified content will be saved.
    list_replace : list
        List of tuples, where each tuple contains the pattern to be replaced
        and its corresponding string.

    Returns
    -------
    bool
        True if the text is replaced successfully, False otherwise.

    Examples
    --------
    >>> original_file = "path/to/original/file.txt"
    >>> target_file = "path/to/target/file.txt"
    >>> replacements = [
    ...     ("pattern1", "replacement1"),
    ...     ("pattern2", "replacement2"),
    ...     ("pattern3", "replacement3")
    ... ]
    >>> success = replace_text(original_file, target_file, replacements)
    >>> if success:
    ...     print("Text replacement successful.")
    ... else:
    ...     print("Text replacement failed.")
    """
    try:
        # Copying the original file to the target file
        shutil.copyfile(original, target)

        # Opening the target file in read and write mode
        with open(target, 'r+') as file:
            # Reading the file data and storing it in a variable
            file_data = file.read()

            # Replacing the patterns with the strings in the file data
            for pattern, replacement in list_replace:
                file_data = file_data.replace(pattern, replacement)

            # Setting the position to the top of the file to overwrite data
            file.seek(0)

            # Writing the replaced data in the file
            file.write(file_data)

            # Truncating the file to remove any extra content
            file.truncate()

        # Return True to indicate successful text replacement
        return True

    except Exception as e:
        print(f"An error occurred while replacing text: {e}")
        return False


def delete_specific_files(folder_path, extensions):
    """
    Delete files with specific extensions from a given folder.

    This function iterates through all the files in the given folder and deletes
    files with extensions specified in the 'extensions' list.

    Parameters
    ----------
    folder_path : str
        Path to the folder containing the files.

    extensions : list of str
        List of file extensions to be deleted.

    Returns
    -------
    None
        The function does not return any value.

    Raises
    ------
    Exception
        If an error occurs while deleting the files.

    Example
    -------
    folder_path = "/path/to/folder"
    extensions_to_delete = ['.log', '.conv', '.h5', '.xdmf']
    delete_specific_files(folder_path, extensions_to_delete)
    """
    try:
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            if os.path.isfile(file_path):
                file_extension = os.path.splitext(filename)[1]
                if file_extension in extensions:
                    os.remove(file_path)
                    print(f"Deleted: {file_path}")
        print("All specified files deleted successfully.")
    except Exception as e:
        print(f"An error occurred: {e}")


def delete_folders_with_contents(folder_path, folder_names):
    """
    Delete specified folders and their contents from a given folder.

    This function iterates through the specified folder names and deletes the
    corresponding folders along with their contents from the given folder.

    Parameters
    ----------
    folder_path : str
        Path to the folder containing the folders to be deleted.

    folder_names : list of str
        List of folder names to be deleted.

    Returns
    -------
    None
        The function does not return any value.

    Raises
    ------
    Exception
        If an error occurs while deleting the folders.

    Example
    -------
    folder_path = "/path/to/folder"
    folder_names = ["folder_to_delete_1", "folder_to_delete_2"]
    delete_folders_with_contents(folder_path, folder_names)
    """

    try:
        for folder_name in folder_names:
            target_folder_path = os.path.join(folder_path, folder_name)
            if os.path.exists(target_folder_path) and os.path.isdir(target_folder_path):
                shutil.rmtree(target_folder_path)
                print(f"Deleted folder and its contents: {target_folder_path}")
        print("All specified folders and their contents deleted successfully.")
    except Exception as e:
        print(f"An error occurred: {e}")


def prepare_simulation(path, results_folder_name="results"):
    """
    Prepare simulation directory by deleting specific files and folders.

    This function prepares a simulation directory by deleting specific files and
    folders that are commonly generated during simulations and creates a folder
    with the specified name.

    Parameters
    ----------
    path : str
        Path to the simulation directory.
    results_folder_name : str, optional
        Name of the folder to be created. Default is "results".

    Returns
    -------
    None
        The function does not return any value.

    Example
    -------
    folder_path = "/path/to/simulation"
    prepare_simulation(folder_path, "my_results")
    """
    extensions_to_delete = ['.log',
                            '.conv',
                            '.h5',
                            '.xdmf',
                            '.energy',
                            '.reaction',
                            '.dof']

    delete_specific_files(path, extensions_to_delete)
    delete_folders_with_contents(path, ["paraview-solutions"])

    results_path = os.path.join(path, results_folder_name)
    delete_folders_with_contents(path, [results_path])
    os.makedirs(results_path)


def delete_folder(folder_path):
    """
    Recursively deletes a folder and its contents.

    Parameters
    ----------
    folder_path : str
        The path to the folder to be deleted.

    Raises
    ------
    OSError
        If an error occurs during the deletion process.

    Notes
    -----
    This function iterates over all items in the specified folder, removing files
    and recursively calling itself for subdirectories. Finally, it removes the
    empty folder.

    Examples
    --------
    >>> delete_folder('/path/to/your/folder')
    Folder '/path/to/your/folder' and its contents deleted successfully.
    """
    try:
        # Iterate over all the items in the folder
        for item in os.listdir(folder_path):
            item_path = os.path.join(folder_path, item)

            # If it's a file, remove it
            if os.path.isfile(item_path):
                os.remove(item_path)
            # If it's a directory, recursively call the function
            elif os.path.isdir(item_path):
                delete_folder(item_path)

        # Finally, remove the empty folder
        os.rmdir(folder_path)
        print(f"Folder '{folder_path}' and its contents deleted successfully.")
    except Exception as e:
        print(f"Error: {e}")


def append_results_to_file(output_file_path, header, step, *data):
    """
    Appends results to a specified file, including a header if the file is empty.

    Parameters
    ----------
    output_file_path : str
        The path to the file where results will be appended.
    header : str
        The header line to write if the file is empty.
    step : int or str
        The step number or identifier to write at the beginning of the line.
    *data : tuple
        Variable length argument list containing the data to append, which will be tab-separated.

    Notes
    -----
    - If the file specified by `output_file_path` is empty, the `header` will be written first.
    - Each subsequent call will append a new line with the `step` and `data` values.

    Examples
    --------
    >>> append_results_to_file('results.txt', 'Time\tValue1\tValue2', 1, 0.5, 1.2)
    >>> append_results_to_file('results.txt', 'Time\tValue1\tValue2', 2, 0.6, 1.3)
    """
    with open(output_file_path, 'a') as file:
        if os.path.getsize(output_file_path) == 0:
            file.write(f'{header}\n')
        data_str = '\t'.join(str(d) for d in data)
        line = f'{step}\t{data_str}\n'
        file.write(line)
