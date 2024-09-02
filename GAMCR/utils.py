from datetime import datetime, timedelta

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