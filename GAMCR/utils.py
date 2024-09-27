from datetime import datetime, timedelta
import numpy as np

def fractional_year_to_datetime(fractional_year):
    # Extract the integer part (year) and the fractional part
    year = int(fractional_year)
    fractional_part = fractional_year - year
    
    # Calculate the number of days in the given year
    start_of_year = datetime(year, 1, 1)
    end_of_year = datetime(year + 1, 1, 1)
    days_in_year = (end_of_year - start_of_year).days
    
    # Calculate the total number of seconds represented by the fractional part
    seconds_in_year = days_in_year * 24 * 3600
    elapsed_seconds = fractional_part * seconds_in_year
    
    # Get the date and time by adding the elapsed seconds to the start of the year
    date_time = start_of_year + timedelta(seconds=elapsed_seconds)
    
    # Round to the nearest hour
    if date_time.minute >= 30:
        date_time += timedelta(hours=1)
    rounded_date_time = date_time.replace(minute=0, second=0, microsecond=0)
    
    return rounded_date_time


def nse(observed, simulated):
    """
    Calculate the Nash-Sutcliffe Efficiency (NSE).
    
    Parameters:
    observed (array-like): Array of observed values.
    simulated (array-like): Array of simulated values.
    
    Returns:
    float: NSE value.
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


def build_custom_matrix(n):
    """
    Build an n x n matrix with the following structure:
    - Diagonal elements: 6
    - Subdiagonal and superdiagonal elements: -4
    - Sub-subdiagonal and super-superdiagonal elements: 1
    
    Parameters:
    - n: Size of the matrix (n x n)
    
    Returns:
    - A numpy array representing the matrix.
    """
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

    matrix[0,:2] = [1,-2]
    matrix[1,:2] = [-2,5]
    matrix[-2,-2:] = [5,-2]
    matrix[-1,-2:] = [-2,1]
    return matrix
