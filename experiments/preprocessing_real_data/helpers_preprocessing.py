import numpy as np
import os
import pandas as pd
import datetime
import sys
sys.path.append(os.path.join('..', '..'))
import pyeto


def toYearFraction(date):
    def sinceEpoch(date):  # returns seconds since epoch
        return (date.timestamp())
    s = sinceEpoch

    year = date.year
    startOfThisYear = datetime.datetime(year=year, month=1, day=1)
    startOfNextYear = datetime.datetime(year=year+1, month=1, day=1)

    yearElapsed = s(date) - s(startOfThisYear)
    yearDuration = s(startOfNextYear) - s(startOfThisYear)
    fraction = yearElapsed/yearDuration

    return date.year + fraction


def get_acronym(serie, dict_acronyms):
    """Return a list of acronyms corresponding to the given series of names.

    This function takes an iterable of names and a dictionary mapping full names
    to their acronyms.

    Args:
        serie (iterable): A sequence (e.g., list, Series) of names to convert.
        dict_acronyms (dict): A dictionary mapping names (keys) to acronyms (values).

    Returns:
        list: A list of acronyms corresponding to the input names.
    """
    res = []
    for name in serie:
        res.append(dict_acronyms[name])

    return res


def get_latitude(geolocator, name_station):
    """Retrieve the latitude of a station using a geolocator service.

    This function uses a geolocator from the `geopy` library to obtain
    the geographic coordinates of a station by its river name.

    Args:
        geolocator: A Nominatim geolocator instance used to perform the
        geocoding.
        name_station (str): The name or address of the station to look up.

    Returns:
        str: The latitude of the station as a string, or 'NAN' if not found.
    """
    location = geolocator.geocode(name_station)
    try:
        return location.raw['lat']
    except:
        print(name_station)
        return np.nan


def get_PET_hargreaves(tmin, tmean, tmax, date, latitude):
    """Compute potential evapotranspiration (PET) using the Hargreaves equation.

    This function estimates daily potential evapotranspiration (PET) following
    the Hargreaves method (Hargreaves & Samani, 1985). It computes
    extraterrestrial radiation from the latitude and day of year and uses
    minimum, mean, and maximum temperature data to estimate PET.

    Args:
        tmin (float): Daily minimum air temperature (°C).
        tmean (float): Daily mean air temperature (°C).
        tmax (float): Daily maximum air temperature (°C).
        date (datetime.date or datetime.datetime): Date of observation.
        latitude (float): Latitude of the location in decimal degrees.

    Returns:
        float: Estimated potential evapotranspiration (mm/day).

    Notes:
        - The function ensures that temperature inputs are physically consistent:
          `tmin ≤ tmean ≤ tmax`.
        - Uses the Hargreaves formulation implemented in the `pyeto` package.
        - Reference: Hargreaves, G.H. & Samani, Z.A. (1985). "Reference crop
          evapotranspiration from temperature." Applied Engineering in Agriculture,
          1(2), 96–99.
    """
    # Convert latitude to radians
    lat = pyeto.deg2rad(float(latitude))

    # Compute day of year and extraterrestrial radiation
    day_of_year = date.timetuple().tm_yday
    sol_dec = pyeto.sol_dec(day_of_year)  # solar declination
    sha = pyeto.sunset_hour_angle(lat, sol_dec)  # sunset hour angle
    ird = pyeto.inv_rel_dist_earth_sun(day_of_year)  # inverse relative distance Earth–Sun
    et_rad = pyeto.et_rad(lat, sol_dec, sha, ird)  # extraterrestrial radiation

    # Ensure temperature consistency
    tmax = max(tmax, tmin)
    tmin = min(tmin, tmax)
    tmean = max(min(tmean, tmax), tmin)

    # Compute PET using Hargreaves equation
    return pyeto.hargreaves(tmin, tmax, tmean, et_rad)


def process_precip(GISID, path_J, path_daily):
    """Prepare and clean precipitation time series for a given catchment.

    This function loads, processes, and fills missing hourly precipitation data
    for a given catchment identified by its GIS ID. It uses both hourly and
    daily datasets to reconstruct missing hourly values when possible.

    Steps:
        1. Identify and load the correct hourly precipitation file.
        2. Parse and sort timestamps, then add a fractional-year variable.
        3. Load daily precipitation data for the same catchment.
        4. Detect and fill gaps in hourly records using daily totals when available.
        5. Remove incomplete periods before and after 2010 if missing days occur.

    Args:
        gis_id (int or str): GIS ID of the catchment to process.
        path_j (str): Path to the folder containing hourly precipitation CSV files.
        path_daily (str): Path to the folder containing daily precipitation data.

    Returns:
        pd.DataFrame: A cleaned and continuous DataFrame with columns:
            - 'datetime': Timestamps of hourly precipitation records.
            - 't': Corresponding time in fractional years.
            - 'precip': Hourly precipitation (mm or equivalent).

    Notes:
        - Missing hourly data are filled proportionally using daily precipitation totals.
        - Days with no daily data are tracked but not filled.
        - Data outside the valid range defined by missing-day periods are removed.
    """
    # Load and prepare hourly precipitation data
    path_fluxes = os.path.join(path_J, f'{GISID}.csv')
    fluxes = pd.read_csv(path_fluxes, sep=",")
    fluxes['datetime'] = fluxes['datetime'].apply(lambda x: datetime.datetime.strptime(str(x), '%Y%m%d%H%M%S'))
    fluxes = fluxes.sort_values(by='datetime', ascending=True)
    fluxes['t'] = np.array([toYearFraction(t) for t in fluxes['datetime']])
    fluxes = fluxes.sort_values(by=['datetime']).reset_index()
    fluxes = fluxes[['mean', 'datetime', 't']]
    fluxes = fluxes.rename(columns={"mean": "precip"})

    # Load daily precipitation data
    daily_path = os.path.join(path_daily, 'sw_rainfall_timeseries', f'psw_{GISID}.csv')
    dayfluxes = pd.read_csv(daily_path, sep=",")
    dayfluxes['datetime'] = pd.to_datetime(dayfluxes['datetime'], format='%Y-%m-%d')

    # Identify gaps larger than one hour
    diffhour = (fluxes['datetime'].diff() > pd.Timedelta(hours=1)).to_numpy()
    idxs = np.where(diffhour > 0.5)[0]

    # Fill missing hourly values using daily totals
    missing_dates_as_day = []
    for idx in idxs:
        current_date_hour = fluxes.iloc[idx-1]['datetime']
        next_date = (fluxes.iloc[idx]['datetime'] + pd.to_timedelta('24 hour')).date()

        while current_date_hour.date() != next_date:
            date = current_date_hour.date()
            day_idx = np.where(dayfluxes['datetime'].apply(lambda x: x.date()) == date)[0]

            if len(day_idx) == 1:
                precipday = dayfluxes.iloc[day_idx[0]]['precip']
                missing_dates_in_hour = []
                date_in_hour = datetime.datetime.strptime(str(date), '%Y-%m-%d')

                for _ in range(24):
                    idx_date_in_hour = np.where(fluxes['datetime'] == date_in_hour)[0]
                    if len(idx_date_in_hour) == 0:
                        missing_dates_in_hour.append(date_in_hour)
                    else:
                        precipday -= fluxes.iloc[idx_date_in_hour[0]]['precip']
                        precipday = max(0, precipday) 
                    date_in_hour = pd.to_timedelta('1 hour') + date_in_hour

                # Fill missing hourly data proportionally
                for date_in_hour in missing_dates_in_hour:
                    idx_date_in_hour = np.where(fluxes['datetime'] == date_in_hour)[0]
                    new_row = {
                        'datetime': date_in_hour,
                        't': toYearFraction(date_in_hour),
                        'precip': precipday / len(missing_dates_in_hour)
                    }
                    fluxes = pd.concat([fluxes, pd.DataFrame([new_row])], ignore_index=True)
            else:
                missing_dates_as_day.append(date)

            current_date_hour = current_date_hour + pd.to_timedelta('24 hour')

    # Final cleanup
    fluxes = fluxes.sort_values(by=['datetime'])
    fluxes = fluxes.reset_index(drop=True)

    # Handle lower limit (before 2010)
    idxlow = np.where([date <= datetime.datetime.strptime('2010-01-01', '%Y-%m-%d').date() for date in missing_dates_as_day])[0]
    if len(idxlow) != 0:
        datelow = np.array(missing_dates_as_day)[idxlow][-1]
        datelow += pd.to_timedelta('24 hour')
        fluxes = fluxes[fluxes['datetime'] >= datetime.datetime.strptime(str(datelow), '%Y-%m-%d')]
        fluxes = fluxes.sort_values(by=['datetime']).reset_index(drop=True)

    # Handle upper limit (after 2010)
    idxup = np.where([date >= datetime.datetime.strptime('2010-01-01', '%Y-%m-%d').date() for date in missing_dates_as_day])[0]
    if len(idxup) != 0:
        dateup = np.array(missing_dates_as_day)[idxup][0]
        dateup -= pd.to_timedelta('24 hour')
        fluxes = fluxes[fluxes['datetime'] <= datetime.datetime.strptime(str(dateup), '%Y-%m-%d')]
        fluxes = fluxes.sort_values(by=['datetime']).reset_index(drop=True)

    return fluxes


def process_hydro(GISID, path_Q, df_catchment_names, path_daily):
    """Prepare and clean discharge time series for a given catchment.

    This function loads, processes, and fills missing hourly discharge data
    for a given catchment identified by its GIS ID. It uses both hourly and
    daily datasets to reconstruct missing hourly values when possible.

    Steps:
        1. Identify the correct hourly discharge file based on the catchment name.
        2. Load and sort the hourly data, converting timestamps and adding a
           fractional-year time variable.
        3. Load corresponding daily discharge data.
        4. Identify and fill gaps in the hourly record using daily data when available.
        5. Remove incomplete data periods before and after 2010 if missing days occur.

    Args:
        gis_id (int or str): GIS ID of the catchment to process.
        path_q (str): Path to the folder containing hourly discharge CSV files.
        df_catchment_names (pd.DataFrame): DataFrame linking GIS IDs to
            catchment names (columns: 'GIS_ID', 'catchment_name').
        path_daily (str): Path to the folder containing daily discharge data.

    Returns:
        pd.DataFrame: A cleaned and continuous DataFrame with columns:
            - 'datetime': Timestamps of hourly discharge records.
            - 't': Corresponding time in fractional years.
            - 'discharge': Hourly discharge (m3/s?).

    Notes:
        - Missing hourly data are filled proportionally using daily discharge totals.
        - Days with no daily data are tracked but not filled.
        - Data outside the valid range defined by missing-day periods are removed.
    """
    # Identify the correct file for the catchment
    files = os.listdir(path_Q)
    cat_name = df_catchment_names.loc[
        df_catchment_names['GIS_ID'] == GISID, 'catchment_name'
    ].values[0]

    matching_files = [f for f in files if cat_name in f]
    if not matching_files:
        raise FileNotFoundError(f"No discharge file found for {cat_name}.")
    file = matching_files[0]

    # Load and process hourly discharge data
    hydro = pd.read_csv(os.path.join(path_Q, file), sep=",")
    hydro['datetime'] = hydro['datetime'].apply(lambda x: datetime.datetime.strptime(str(x), '%Y-%m-%d %H:%M:%S'))
    hydro = hydro.sort_values(by='datetime', ascending=True)
    hydro['t'] = np.array([toYearFraction(t) for t in hydro['datetime']])
    hydro = hydro.sort_values(by=['datetime'])
    hydro = hydro.reset_index()
    hydro = hydro[['cms', 'datetime', 't']]
    hydro = hydro.rename(columns={"cms": "discharge"})

    # Load daily discharge data
    daily_path = os.path.join(path_daily, 'sw_hydrographs', f'sw_{GISID}.csv')
    dayhydro = pd.read_csv(daily_path, sep=",")
    dayhydro['datetime'] = pd.to_datetime(dayhydro['datetime'], format='%Y-%m-%d')

    # Identify gaps larger than 1 hour
    diffhour = (hydro['datetime'].diff() > pd.to_timedelta('1 hour')).to_numpy()
    idxs = np.where(diffhour > 0.5)[0]

    missing_dates_as_day = []
    for idx in idxs:
        current_date_hour = hydro.iloc[idx-1]['datetime']
        next_date = (hydro.iloc[idx]['datetime'] + pd.Timedelta(hours=24)).date()

        while current_date_hour.date() != next_date:
            date = current_date_hour.date()
            day_idx = np.where(dayhydro['datetime'].apply(lambda x: x.date()) == date)[0]

            if len(day_idx) == 1:
                dischargeday = dayhydro.iloc[day_idx[0]]['discharge']
                missing_dates_in_hour = []
                date_in_hour = datetime.datetime.strptime(str(date), '%Y-%m-%d')

                for _ in range(24):
                    idx_date_in_hour = np.where(hydro['datetime'] == date_in_hour)[0]
                    if len(idx_date_in_hour) == 0:
                        missing_dates_in_hour.append(date_in_hour)
                    else:
                        dischargeday -= hydro.iloc[idx_date_in_hour[0]]['discharge']
                        dischargeday = max(0, dischargeday)
                    date_in_hour = pd.to_timedelta('1 hour') + date_in_hour

                # Fill missing hourly data proportionally
                for date_in_hour in missing_dates_in_hour:
                    idx_date_in_hour = np.where(hydro['datetime'] == date_in_hour)[0]
                    new_row = {
                        'datetime': date_in_hour,
                        't': toYearFraction(date_in_hour),
                        'discharge': dischargeday / len(missing_dates_in_hour)
                    }
                    hydro = pd.concat([hydro, pd.DataFrame([new_row])], ignore_index=True)
            else:
                missing_dates_as_day.append(date)

            current_date_hour += pd.Timedelta(hours=24)

    # Final cleanup
    hydro = hydro.sort_values(by=['datetime'])
    hydro = hydro.reset_index(drop=True)

    # Handle lower limit (before 2010)
    idxlow = np.where([date <= datetime.datetime.strptime('2010-01-01', '%Y-%m-%d').date() for date in missing_dates_as_day])[0]
    if len(idxlow) != 0:
        datelow = np.array(missing_dates_as_day)[idxlow][-1]
        datelow += pd.Timedelta(hours=24)
        hydro = hydro[hydro['datetime'] >= datetime.datetime.strptime(str(datelow), '%Y-%m-%d')]
        hydro = hydro.sort_values(by=['datetime']).reset_index(drop=True)

    # Handle upper limit (after 2010)
    idxup = np.where([date >= datetime.datetime.strptime('2010-01-01', '%Y-%m-%d').date() for date in missing_dates_as_day])[0]
    if len(idxup) != 0:
        dateup = np.array(missing_dates_as_day)[idxup][0]
        dateup -= pd.Timedelta(hours=24)
        hydro = hydro[hydro['datetime'] <= datetime.datetime.strptime(str(dateup), '%Y-%m-%d')]
        hydro = hydro.sort_values(by=['datetime']).reset_index(drop=True)

    return hydro


def merging_dfs(fluxes, hydro):
    """
    Merging the dataset
    """
    mindate = hydro['datetime'].to_numpy()[0]
    mindate = max(mindate, fluxes['datetime'].to_numpy()[0])

    maxdate = hydro['datetime'].to_numpy()[-1]
    maxdate = min(maxdate, fluxes['datetime'].to_numpy()[-1])
    hydro = hydro[hydro['datetime'] >= mindate]
    hydro = hydro[hydro['datetime'] <= maxdate]
    hydro = hydro.sort_values(by=['datetime'])
    hydro = hydro.reset_index(drop=True)

    fluxes = fluxes[fluxes['datetime'] >= mindate]
    fluxes = fluxes[fluxes['datetime'] <= maxdate]
    fluxes = fluxes.sort_values(by=['datetime'])
    fluxes = fluxes.reset_index(drop=True)

    dfdata = {'discharge': hydro['discharge'].to_numpy(),
              'precip': fluxes['precip'].to_numpy(),
              't': fluxes['t'].to_numpy(),
              'datetime': fluxes['datetime'].to_numpy()}
    dfdata = pd.DataFrame(dfdata)

    return dfdata
