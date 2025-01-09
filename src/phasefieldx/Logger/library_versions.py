"""
Logger/library versions
=======================

This functions provides utility functions for logging simulation information and system details using Python's logging module.
It also logs versions of important libraries such as Python, DolfinX, ufl, basix, numpy, and logging, along with system
information like platform details and Python version.

"""

import phasefieldx
import logging
import os
import sys
import dolfinx
import numpy as np
import ufl
import time
import basix
import platform


def set_logger(result_folder_name):
    """
    Set up a logger for logging simulation information.

    Parameters
    ----------
    result_folder_name : str
        The name of the folder where the log file will be stored.

    Returns
    -------
    logging.Logger
        The logger object set up for logging simulation information.

    Notes
    -----
    This function sets up a logger named 'simulation_logger' with INFO level logging.
    It creates a log file named 'simulation.log' in the specified result folder.

    Examples
    --------
    >>> logger = set_logger('results_folder')
    """
    logger = logging.getLogger('simulation_logger')
    logger.setLevel(logging.INFO)
    simulation_file_handler = logging.FileHandler(
        os.path.join(result_folder_name, 'simulation.log'))
    simulation_formatter = logging.Formatter('%(message)s')
    simulation_file_handler.setFormatter(simulation_formatter)
    logger.addHandler(simulation_file_handler)
    return logger


def log_library_versions(logger):
    """
    Log versions of important libraries.

    Parameters
    ----------
    logger : logging.Logger
        The logger object to log the information.

    Returns
    -------
    None

    Notes
    -----
    This function logs the versions of important libraries including Python, DolfinX,
    ufl, basix, numpy, and logging.
    """
    logger.info(f"Python version: {sys.version_info.major}.{sys.version_info.minor}")
    logger.info("=========== Library Versions ===========")
    logger.info(f"PhaseFieldX : {phasefieldx.__version__}")
    logger.info(f"DolfinX : {dolfinx.__version__}")
    logger.info(f"ufl : {ufl.__version__}")
    logger.info(f"basix : {basix.__version__}")
    logger.info(f"numpy : {np.__version__}")
    logger.info(f"logging : {logging.__version__}")
    logger.info("=======================================")


def log_system_info(logger):
    """
    Log system information using the provided logger.

    Parameters
    ----------
    logger : logging.Logger
        The logger object to be used for logging system information.

    Returns
    -------
    None

    Notes
    -----
    This function logs various system information including operating system,
    architecture, user name, processor, machine type, and Python version.
    """
    logger.info("=========== Platform ==================")
    logger.info(f"Operating System Information: {platform.platform()}")
    logger.info(f"Architecture : {platform.architecture()}")
    logger.info(f"User name : {platform.uname()}")
    logger.info(f"processor : {platform.processor()}")
    logger.info(f"Machine type : {platform.machine()}")
    logger.info(f"Python version : {platform.python_version()}")
    logger.info("=======================================")


def log_end_analysis(logger, totaltime=0.0):
    logger.info(f"\n\n\n ====================================================")
    logger.info(f"\n\n End of computations")
    logger.info(f" Analysis finished correctly.")
    logger.info(f" total simulation time: {totaltime}")
    logger.info(f"Analysis finished on {time.strftime('%a %b %d %H:%M:%S %Y', time.localtime())}")


def log_model_information(msh, logger):
    logger.info("========== Mesh Information ===========")
    msh.geometry.dim
    logger.info(f"Dimension: {msh.geometry.dim}")
