from datetime import datetime, timedelta
import numpy as np


def fractional_year_to_datetime(fractional_year):
    """Convert a fractional (decimal) year to a datetime object.

    The function translates a fractional year (e.g., 2015.75) into the
    corresponding calendar date and time, rounded to the nearest hour.
    Leap years are handled correctly.

    Parameters
    ----------
    fractional_year : float
        Year expressed as a floating-point number, where the integer part
        is the year (e.g., 2015) and the fractional part represents the
        portion of the year that has elapsed.

    Returns
    -------
    datetime.datetime
        Datetime object representing the corresponding calendar date,
        rounded to the nearest hour.
    """
    # Separate the integer year and fractional component
    year = int(fractional_year)
    fractional_part = fractional_year - year

    # Compute number of days in the year (accounts for leap years)
    start_of_year = datetime(year, 1, 1)
    end_of_year = datetime(year + 1, 1, 1)
    days_in_year = (end_of_year - start_of_year).days

    # Convert fractional year into elapsed seconds
    seconds_in_year = days_in_year * 24 * 3600
    elapsed_seconds = fractional_part * seconds_in_year

    # Compute the full datetime
    date_time = start_of_year + timedelta(seconds=elapsed_seconds)

    # Round to nearest hour
    if date_time.minute >= 30:
        date_time += timedelta(hours=1)
    rounded_date_time = date_time.replace(minute=0, second=0, microsecond=0)

    return rounded_date_time


def nse(observed, simulated):
    """
    Calculate the Nash-Sutcliffe Efficiency (NSE).

    Parameters
    ----------
        observed : np.ndarray
            Array of observed values.
        simulated : np.ndarray
            Array of simulated values.

    Returns
    -------
        float
            NSE value.
    """
    observed = np.array(observed)
    simulated = np.array(simulated)

    # Calculate the mean of the observed data
    mean_observed = np.mean(observed)

    # Compute the numerator and denominator of the NSE formula
    numerator = np.sum((observed - simulated) ** 2)
    denominator = np.sum((observed - mean_observed) ** 2)

    # Compute NSE
    nse_value = 1 - (numerator / denominator)

    return nse_value


def compute_smoothing_penalty_matrix(n, knots):
    # Create an n x n zero matrix
    matrix = np.zeros((n, n))

    # Set the diagonal to 6
    np.fill_diagonal(matrix, 6)

    # Set the subdiagonal and superdiagonal to -4
    np.fill_diagonal(matrix[1:], -4)
    np.fill_diagonal(matrix[:, 1:], -4)

    # Set the sub-subdiagonal and super-superdiagonal to -1
    np.fill_diagonal(matrix[2:], 1)
    np.fill_diagonal(matrix[:, 2:], 1)

    matrix[0, :2] = [1, -2]
    matrix[1, :2] = [-2, 5]
    matrix[-2, -2:] = [5, -2]
    matrix[-1, -2:] = [-2, 1]

    d_knots = np.diff(knots)
    matrix = matrix * (d_knots.reshape(-1, 1) @ d_knots.reshape(1, -1))

    return matrix


def build_custom_matrix(n):
    """
    Build an n x n matrix with the following structure:
    - Main diagonal elements: 6
    - Subdiagonal and superdiagonal elements: -4
    - Sub-subdiagonal and super-superdiagonal elements: 1

    Parameters
    ----------
    n : int
        Size of the square matrix.

    Returns
    -------
    np.ndarray
        Symmetric (n, n) matrix with the specified structure.
    """
    # Initialize an n×n zero matrix
    matrix = np.zeros((n, n))

    # Set the diagonal to 6
    np.fill_diagonal(matrix, 6)

    # Set the subdiagonal and superdiagonal to -4
    np.fill_diagonal(matrix[1:], -4)
    np.fill_diagonal(matrix[:, 1:], -4)

    # Set the sub-subdiagonal and super-superdiagonal to -1
    np.fill_diagonal(matrix[2:], 1)
    np.fill_diagonal(matrix[:, 2:], 1)

    # Adjust boundary rows for stability / boundary conditions
    matrix[0, :2] = [1, -2]
    matrix[1, :2] = [-2, 5]
    matrix[-2, -2:] = [5, -2]
    matrix[-1, -2:] = [-2, 1]

    return matrix
