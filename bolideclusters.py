from bolides import BolideDataFrame
import numpy as np
from sklearn.metrics import DistanceMetric
from sklearn.neighbors import BallTree
from bolides import fov_utils as fu
import pandas as pd
from tqdm import tqdm 
import numpy as np
from scipy.special import comb
from shapely.geometry import Point
import pyproj
from pyproj import Transformer
import matplotlib.pyplot as plt
import scipy.stats
from scipy.interpolate import interp1d, PchipInterpolator
from bolides import ShowerDataFrame
from scipy.stats import binned_statistic
from scipy.ndimage import gaussian_filter1d


fov = None  # global variable
min_time = None
max_time = None
total_time = None
n = None # number of detections

def initialize(boundary, bdf_final):
    """Initialize constant variables"""
    global fov
    global min_time
    global max_time
    global total_time
    global n
    n = len(bdf_final)
    poly = fu.get_boundary(boundary, collection=True, intersection=False, crs=None)
    fov = poly.area / 1e6  # km^2
    min_time = min(bdf_final['datetime'].to_list())
    max_time = max(bdf_final['datetime'].to_list())
    total_time = (max_time - min_time).total_seconds()
    return fov, min_time, max_time, total_time, n
    
def make_bdf(boundary):
    """
    Creates a dataframe of bolide data with columns: time in seconds (from the earliest observation), latitude, longitude (in degrees)
    
    Parameters
    ----------
    boundary: str or iterable with multiple strings
        Specifies the boundaries desired.
    - ``'goes'``: Combined FOV of the GLM aboard GOES-16 and GOES-17
    - ``'goes-w'``: GOES-West position GLM FOV, currently corresponding to GOES-17.
      Note that this combines the inverted and non-inverted FOVs.
    - ``'goes-e'``: GOES-East position GLM FOV, currently corresopnding to GOES-16.
    - ``'goes-w-ni'``: GOES-West position GLM FOV, when GOES-17 is not inverted (summer).
    - ``'goes-w-i'``: GOES-West position GLM FOV, when GOES-17 is inverted (winter).
    - ``'goes-17-89.5'``: GOES-17 GLM FOV when it was in its checkout orbit.
    - ``'fy4a'``: Combined FOV of the Fengyun-4A LMI, in both North and South configurations.
    - ``'fy4a-n'``: Fengyun-4A LMI FOV when in the North configuration (summer).
    - ``'fy4a-s'``: Fengyun-4A LMI FOV when in the South configuration (winter).
    - ``'gmn-25km'``: Combined FOV of all Global Meteor Network stations at 25km detection altitude.
    - ``'gmn-70km'``: Combined FOV of all Global Meteor Network stations at 70km detection altitude.
    - ``'gmn-100km'``: Combined FOV of all Global Meteor Network stations at 100km detection altitude.

    Returns
    -------
    `~BolideDataFrame`
            The filtered `~BolideDataFrame`
    """
    bdf = BolideDataFrame(source='glm')
    bdf_boundary = bdf.filter_boundary(boundary=[boundary])
    # bdf_boundary.add_website_data()
    bdf_boundary = bdf_boundary[['datetime', 'latitude', 'longitude']]
    bdf_final = bdf_boundary.copy()
    bdf_final['time_seconds'] = (bdf_boundary.iloc[:, 0] - bdf_boundary.iloc[:, 0].min()).dt.total_seconds()
    bdf_final = bdf_final[['time_seconds', 'latitude', 'longitude', 'datetime']]
    return bdf_final

def make_bdf_showers(bdf, sdf, n):
    """
    Creates a dataframe of bolide data filtering out n showers with columns: time in seconds (from the earliest observation), latitude, longitude (in degrees)
    
    Parameters
    ----------
    boundary: str or iterable with multiple strings
        Specifies the boundaries desired.
    - ``'goes'``: Combined FOV of the GLM aboard GOES-16 and GOES-17
    - ``'goes-w'``: GOES-West position GLM FOV, currently corresponding to GOES-17.
      Note that this combines the inverted and non-inverted FOVs.
    - ``'goes-e'``: GOES-East position GLM FOV, currently corresopnding to GOES-16.
    - ``'goes-w-ni'``: GOES-West position GLM FOV, when GOES-17 is not inverted (summer).
    - ``'goes-w-i'``: GOES-West position GLM FOV, when GOES-17 is inverted (winter).
    - ``'goes-17-89.5'``: GOES-17 GLM FOV when it was in its checkout orbit.
    - ``'fy4a'``: Combined FOV of the Fengyun-4A LMI, in both North and South configurations.
    - ``'fy4a-n'``: Fengyun-4A LMI FOV when in the North configuration (summer).
    - ``'fy4a-s'``: Fengyun-4A LMI FOV when in the South configuration (winter).
    - ``'gmn-25km'``: Combined FOV of all Global Meteor Network stations at 25km detection altitude.
    - ``'gmn-70km'``: Combined FOV of all Global Meteor Network stations at 70km detection altitude.
    - ``'gmn-100km'``: Combined FOV of all Global Meteor Network stations at 100km detection altitude.

    Returns
    -------
    `~BolideDataFrame`
            The filtered `~BolideDataFrame`
    """

    # filter fastest showers
    codes = sdf.fastest_showers(n)['Code'].values
    if n == 0:
        bdf_boundary = bdf.copy()
    else:
        bdf_boundary = bdf.filter_shower(codes, exclude=True)
    
    bdf_boundary = bdf_boundary[['datetime', 'latitude', 'longitude']]
    bdf_final = bdf_boundary.copy()
    bdf_final['time_seconds'] = (bdf_boundary.iloc[:, 0] - bdf_boundary.iloc[:, 0].min()).dt.total_seconds()
    bdf_final = bdf_final[['time_seconds', 'latitude', 'longitude', 'datetime']]
    return bdf_final


def haversine(lat1, lon1, lat2, lon2):
    """
    Calculates the haversine distance between two points in km.
    Inputs are in degrees.
    """
    R = 6371  # Earth radius in km

    # Convert degrees to radians
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    c = 2 * np.arcsin(np.sqrt(a))

    return R * c



def probability(bdf_final, ind):
    """
    Calculates the likeliness of the identified bolides coming from the same object. ind is a pair of detections [j, k]
    """
    t1 = bdf_final['time_seconds'][ind[0]]
    t2 = bdf_final['time_seconds'][ind[1]]
    lat1 = bdf_final['latitude'][ind[0]]
    lat2 = bdf_final['latitude'][ind[1]]
    long1 = bdf_final['longitude'][ind[0]]
    long2 = bdf_final['longitude'][ind[1]]
    dist = haversine(lat1, long1, lat2, long2)
    dt = abs(t1 - t2)
    
    return (np.pi*dist**2/fov)*(2*dt/total_time) * comb(n, 2, exact=False)
    
def distance_test1(x, y, fov=fov, total_time=total_time):
    """
    Calculates the distance between two points considering time
    ----------------------------
    Inputs: x, y = row i, row j that have columns datetime, lat, long
    Outputs: DistanceMetric64 "distance" between the two points 
    """
    t1, x1, y1 = x
    t2, x2, y2 = y
    spatial_dist = haversine(x1, y1, x2, y2) 
    dt = abs(t1 - t2)
    return (np.pi*dist**2/fov)*(2*dt/total_time) * comb(n, 2, exact=False)

def distance_test2(x, y, space_weight=1.0, time_weight=1.0):
    """
    Calculates the distance between two points considering time
    ----------------------------
    Inputs: x, y = row i, row j that have columns datetime, lat, long
    Outputs: DistanceMetric64 "distance" between the two points 
    """
    
    t1, lat1, lon1 = x
    t2, lat2, lon2 = y

    spatial_dist = haversine(lat1, lon1, lat2, lon2)  # in km
    time_dist = abs(t1 - t2) / 3600  # in hours

    return space_weight * spatial_dist# + time_weight * time_dist

def delete_duplicates(ind):
    """
    # if the closest pair is the same for both objects, get rid of the duplicate index showing that
    """
    delete = []
    for i in range(len(ind)):
        pair = ind[i][1]
        if np.array_equiv(ind[i], np.asarray([ind[pair][1], ind[pair][0]])): # if the opposite is in array, mark the first showing of it
            if ind[i][1] > ind[i][0]: # avoid marking duplicates
                delete.append(i)
            else:
                pass
        else:
            pass
    delete_descend = delete[::-1] # don't mess up order of indices
    return delete_descend

def delete_far(sorted_distances, sorted_pairs, bdf_final):
    """
    input: ind: list of index pairs, bdf_final: bdf with only time seconds, lat, long
    deletes pairs that are far from each other, only left with those that are close in time and distance
    """
    keep = []
    for i in range(len(sorted_pairs)):
        t1, x1, y1 = bdf_final.iloc[sorted_pairs[i][0]]
        t2, x2, y2 = bdf_final.iloc[sorted_pairs[i][1]]
        hav = haversine(x1,y1,x2,y2)
        dt = abs(t1-t2) 
        if hav <= 10 and dt <= 3600:
            keep.append(i)

        if hav > 10 and dt > 3600: # too far apart

            break
    sublist = []
    sub = []
    for i in keep:
        sublist.append(sorted_pairs[i]) # sorted pairs
        sub.append(sorted_distances[i]) # sorted distances that meet criteria
    return sublist, sub
    
def run_tree_pairs(bdf_final, distance_metric):
    """
    Calculates nearest neighbors. Input the distance metric. Returns inds and dists in order 
    """
    bdf_tree = bdf_final[['time_seconds', 'latitude', 'longitude']]
    X = bdf_tree.to_numpy()
    tree = BallTree(X, metric='pyfunc', func=distance_metric)
    

    # Assume `X` is your data and `tree` is the BallTree
    results = []
    batch_size = 500
    for i in tqdm(range(0, len(X), batch_size), desc="Querying"):
        batch = X[i:i + batch_size]
        dists, inds = tree.query(batch, k=2)  # or whatever k you want
        results.append((dists, inds))
    dist = np.vstack([d for d, _ in results])
    ind  = np.vstack([i for _, i in results])
    delete_descend = delete_duplicates(ind)
    ind_filtered = list(ind)
    dist_filtered = list(dist)
    for i in delete_descend:
        ind_filtered.pop(i)
        dist_filtered.pop(i)
    dist_final = [d[1] for d in dist_filtered] # just the dist, not in form [0, dist]
    pairs_distances = list(zip(dist_final, ind_filtered))
    pairs_distances.sort(key=lambda x: x[0])
    sorted_distances, sorted_pairs = zip(*pairs_distances)

    # Convert to list
    sorted_distances = list(sorted_distances)
    sorted_pairs = list(sorted_pairs)
    return sorted_distances, sorted_pairs

#######################################################
# for the CDFs
#######################################################

# AEQD projection (same as your polygon)
aeqd_crs = pyproj.CRS(proj='aeqd', ellps='WGS84', datum='WGS84', lat_0=90, lon_0=0)
transformer = Transformer.from_crs("EPSG:4326", aeqd_crs, always_xy=True)
poly_west = fu.get_boundary('goes-w', collection=True, intersection=False, crs=None)
poly_east = fu.get_boundary('goes-e', collection=True, intersection=False, crs=None)


def create_lonlat(bdf_final):
    """
    Create a numpy arrays with latitude and longitude and sort them by least to greatest
    """
    bdf_lat = bdf_final['latitude'].to_numpy()
    bdf_lon = bdf_final['longitude'].to_numpy()
    return bdf_lon, bdf_lat

def in_poly(lon, lat, poly, transformer=transformer):
    """checks if in polygon"""
    # Project the point to AEQD
    x, y = transformer.transform(lon, lat)
    projected_point = Point(x, y)
    
    # Now test if it's within the polygon
    is_inside = projected_point.within(poly)
    return is_inside

def in_stereo(lon, lat, transformer=transformer):
    """checks if long/lat is in stereo region"""
    # Project the point to AEQD
    x, y = transformer.transform(lon, lat)
    projected_point = Point(x, y)
    
    # Now test if it's within the polygon
    if projected_point.within(poly_west) and projected_point.within(poly_east):
        return True
    else:
        return False
    

def montecarlo(len_bdf, poly, stereo=False, just_stereo=False):
    """
    Monte Carlo simulation to estimate the probability of a bolide being in the FOV.

    Parameters
    ----------
    len_bdf: int
        The number of bolides in the bolide data frame.
    stereo: bool, optional
        If True, samples will be generated within the entire polygon. Default is False.
        If False, samples will be generated only within the non-stereo region of the polygon.
    poly: shapely.geometry.Polygon
        The polygon defining the FOV boundary.
    just_stereo: bool, optional
        If True, only samples within the stereo region will be generated. Default is False.
    """

    accepted_lats = []
    accepted_lons = []
    if just_stereo and not stereo:
        raise ValueError("just_stereo can only be True if stereo is also True.")

    while len(accepted_lats) < len_bdf: # uncomment to use the same number of samples as bolides
        lat = np.random.uniform(-90, 90)
        lon = np.random.uniform(-180, 180)
        if just_stereo: # only those in stereo region and within poly
            if in_poly(lon, lat, poly) and in_stereo(lon, lat):
                accepted_lats.append(lat)
                accepted_lons.append(lon)
        elif stereo: # all those in poly, regardless of stereo region
            if in_poly(lon, lat, poly):
                accepted_lats.append(lat)
                accepted_lons.append(lon)
        else: # all those in non-stereo region
            if in_poly(lon, lat, poly) and not in_stereo(lon, lat):
                accepted_lats.append(lat)
                accepted_lons.append(lon)

    return np.array(accepted_lons), np.array(accepted_lats)

def cdf(x):
    """
    Calculates the empiricle cumulative distribution function (eCDF) of a given array.
    
    Parameters
    ----------
    x : array-like
        The input data for which to compute the CDF.
        
    Returns
    -------
    tuple
        A tuple containing the sorted unique values of `x` and their corresponding CDF values.
    """
    x = np.asarray(x)
    sorted_x = np.sort(x)
    cdf_values = np.arange(1, len(sorted_x) + 1) / len(sorted_x)
    
    return sorted_x, cdf_values

def wrap_lon(lon):
    """
    Wraps longitude values to the range [0, 360).

    Parameters
    ----------
    lon : array-like
        The input longitude values.

    Returns
    -------
    array-like
        The wrapped longitude values.
    """
    lon = np.asarray(lon)
    return (lon + 360) % 360


def wrap_lat(lat):
    """
    Wraps latitude values to the range [0, 180).

    Parameters
    ----------
    lat : array-like
        The input latitude values.

    Returns
    -------
    array-like
        The wrapped latitude values, ensuring they are within the range [0, 180).
    """
    lat = np.asarray(lat)
    return (lat + 180) % 180


def lon_cdf(x, y, x1, y1, title):
    """
    Plots the cumulative distribution function (CDF) of a given array.

    Parameters
    ----------
    x : array-like
        The observed longitude data for which to compute the CDF.
    y : array-like
        The CDF values corresponding to `x`.
    x1 : array-like
        The simulated longitude values.
    y1 : array-like
        The CDF values corresponding to the simulated longitude values.
    title : str, optional
        Additional info for the title of the plot. Like if it includes stereo or not and if it's GOES-East or GOES-West.
    """


    plt.figure(figsize=(10, 6), dpi=150)
    # Use colorblind-friendly colors
    # Blue: #0072B2, Orange: #E69F00, Red: #D55E00, Green: #009E73, Purple: #CC79A7
    plt.plot(x, y, marker='.', linestyle='-', color='#0072B2', label='Observed')
    plt.plot(x1, y1, marker='.', linestyle='none', color='#CC79A7', label='Simulated')
    plt.xlabel('Longitude (degrees)')
    plt.ylabel('CDF')
    plt.title('CDF of Longitude: ' + title)
    plt.grid()
    plt.legend()
    plt.show()
    
def lat_cdf(x, y, x1, y1, title):
    """
    Plots the cumulative distribution function (CDF) of a given array.

    Parameters
    ----------
    x : array-like
        The observed latitude data for which to compute the CDF.
    y : array-like
        The CDF values corresponding to `x`.
    x1 : array-like
        The simulated latitude values.
    y1 : array-like
        The CDF values corresponding to the simulated latitude values.
    title : str, optional
        Additional info for the title of the plot. Like if it includes stereo or not and if it's GOES-East or GOES-West.
    """

    plt.figure(figsize=(10, 6), dpi=150)
    # Use colorblind-friendly colors
    # Blue: #0072B2, Orange: #E69F00, Red: #D55E00, Green: #009E73, Purple: #CC79A7
    plt.plot(x, y, marker='.', linestyle='-', color='#0072B2', label='Observed')
    plt.plot(x1, y1, marker='.', linestyle='-', color='#CC79A7', label='Simulated')
    plt.xlabel('Latitude (degrees)')
    plt.ylabel('CDF')
    plt.title('CDF of Latitude: ' + title)
    plt.grid()
    plt.legend()
    plt.show()


def compare_pvals_showers():
    """
    Compares the CDFs of latitude and longitude for GOES and changes the number of showers filtered out. highest Vg showers filtered first

    Outputs
    -------
    - Plots Pvalue for each ks-test comparison of latitudes and longitudes
    """
    boundary = 'goes'
    all_pvalues_lat = []
    all_pvalues_lon = []
    bdf = BolideDataFrame(source='glm')
    # Filter out dates before July 1, 2019
    bdf_boundary = bdf.filter_boundary(boundary='goes')
    bdf_boundary = bdf_boundary[bdf_boundary['datetime'] >= pd.Timestamp('2019-07-01', tz='UTC')]
    
    sdf = ShowerDataFrame()
    for i in tqdm(range(0, 61)):
        if i == 0:
            bdf = make_bdf_showers(bdf_boundary, sdf, i)
            lon, lat = create_lonlat(bdf)
            lat, lat_cdf_vals = cdf(lat)
            len_bdf = len(lat)
            # Convert longitudes to [0, 360) for CDF
            lon_360 = wrap_lon(lon)
            lon_360, lon_360_cdf = cdf(lon_360)
        else:
            lon_360 = lon_3601
            lat = lat1


        bdf1 = make_bdf_showers(bdf_boundary, sdf, i+2)
        lon1, lat1 = create_lonlat(bdf1)
        lat1, lat_cdf_vals1 = cdf(lat1)
        len_bdf1 = len(lat1)
        # Convert longitudes to [0, 360) for CDF
        lon_3601 = wrap_lon(lon1)
        lon_3601, lon_360_cdf1 = cdf(lon_3601)

        # Calculate the KS test
        # cdf_lat = interp1d(lat_sim, lat_sim_cdf, bounds_error=False, fill_value=(0.0, 1.0))
        # cdf_lon = interp1d(sim_lon_360, sim_lon_360_cdf, bounds_error=False, fill_value=(0.0, 1.0))
        # ks_lat = scipy.stats.ks_1samp(lat, cdf_lat)
        ks_lat = scipy.stats.ks_2samp(lat, lat1)
        pvalue_lat = ks_lat[1]
        # ks_lon = scipy.stats.ks_1samp(lon_360, cdf_lon)
        ks_lon = scipy.stats.ks_2samp(lon_360, lon_3601)
        pvalue_lon = ks_lon[1]
        all_pvalues_lat.append(pvalue_lat)
        all_pvalues_lon.append(pvalue_lon)
        
    all_indices = list(range(0, 61))
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), dpi=300)

    # Latitude CDF
    axes[0].plot(all_indices, all_pvalues_lat, marker='.', linestyle='-')
    axes[0].set_xlabel('Number of showers masked', fontsize=17)
    axes[0].set_ylabel('p-value', fontsize=17)
    axes[0].set_title('Change in Latitude CDF', fontsize=18)
    axes[0].tick_params(axis='both', labelsize=15)
    axes[0].grid()
    axes[0].legend(fontsize=17)

    # Longitude CDF
    axes[1].plot(all_indices, all_pvalues_lon, marker='.', linestyle='-')
    axes[1].set_xlabel('Number of showers masked', fontsize=17)
    axes[1].set_ylabel('p-value', fontsize=17)
    axes[1].set_title('Change in Longitude CDF', fontsize=18)
    axes[1].tick_params(axis='both', labelsize=15)
    axes[1].grid()
    axes[1].legend(fontsize=17)


    plt.tight_layout()
    plt.show()
    return all_pvalues_lat, all_pvalues_lon

def run_sims(boundary, stereo=False, just_stereo=False):
    """
    runs 100 simulations for bolide observations

    Parameters
    ----------
    boundary : str
        The boundary to use for the CDF plot, typically 'goes-w' or 'goes-e'.
    title : str
        The title for the plot, typically indicating the boundary and whether stereo mode is used.
    stereo : bool, optional
        Whether to use stereo mode (default is False).
    just_stereo : bool, optional
        If True, only samples within the stereo region will be used (default is False).

    Returns
    ----------


    """
    if boundary == 'goes-w':
        poly = poly_west
    elif boundary == 'goes-e':
        poly = poly_east
    elif boundary == 'goes':
        poly = fu.get_boundary('goes', collection=True, intersection=False, crs=None)
        
    else:
        raise ValueError("Unsupported boundary. Use 'goes-w', 'goes-e', or 'goes'.")
    bdf = BolideDataFrame(source='glm')
    # Filter out dates before July 1, 2019
    bdf_boundary = bdf.filter_boundary(boundary=boundary)
    bdf_boundary = bdf_boundary[bdf_boundary['datetime'] >= pd.Timestamp('2019-07-01', tz='UTC')]
    
    sdf = ShowerDataFrame()
    bdf_final = make_bdf_showers(bdf_boundary, sdf, 41)
            

    lon, lat = create_lonlat(bdf_final)

    if not stereo:
        # Filter out latitudes and longitudes that are in the stereo region
        accepted_lats = []
        accepted_lons = []

        for lat, lon in zip(lat, lon):
            if not in_stereo(lon, lat):
                accepted_lats.append(lat)
                accepted_lons.append(lon)
    elif just_stereo:
        # Filter out latitudes and longitudes that are not in the stereo region
        accepted_lats = []
        accepted_lons = []

        for lat, lon in zip(lat, lon):
            if in_stereo(lon, lat):
                accepted_lats.append(lat)
                accepted_lons.append(lon)
    else:
        # Use all latitudes and longitudes in the stereo region
        accepted_lats = lat
        accepted_lons = lon


    lat, lat_cdf_vals = cdf(accepted_lats)
    len_bdf = len(accepted_lats)
    lon_360 = wrap_lon(accepted_lons)
    lon_360, lon_360_cdf = cdf(lon_360)

    all_lon_sim = []
    all_lat_sim = []
    all_lon_sim_cdf = []
    all_lat_sim_cdf = []
    pvals_lon = []
    pvals_lat = []
    # Prepare simulated latitudes/longitudes in the stereo region
    for i in tqdm(range(0, 99)):
        lon_samples, lat_samples = montecarlo(len_bdf, poly=poly, stereo=stereo, just_stereo=just_stereo)

        lat_sim, lat_sim_cdf = cdf(lat_samples)

        # Convert longitudes to [0, 360) for CDF
        sim_lon_360 = wrap_lon(lon_samples)
        sim_lon_360, sim_lon_360_cdf = cdf(sim_lon_360)

        all_lon_sim.append(sim_lon_360)
        all_lat_sim.append(lat_sim)
        all_lon_sim_cdf.append(sim_lon_360_cdf)
        all_lat_sim_cdf.append(lat_sim_cdf)


        # ks_lat = scipy.stats.ks_1samp(lat, cdf_lat)
        ks_lat = scipy.stats.ks_2samp(accepted_lats, lat_sim)
        pvalue_lat = ks_lat[1]
        # ks_lon = scipy.stats.ks_1samp(lon_360, cdf_lon)
        ks_lon = scipy.stats.ks_2samp(lon_360, sim_lon_360)
        pvalue_lon = ks_lon[1]

        pvals_lat.append(pvalue_lat)
        pvals_lon.append(pvalue_lon)

    return lon_360, lon_360_cdf, all_lon_sim, all_lon_sim_cdf, pvals_lon, lat, lat_cdf_vals, all_lat_sim, all_lat_sim_cdf, pvals_lat

def interp(all_sim, all_sim_cdf, real_x):
    # interpolate
    num_sim = len(all_sim)
    x_common = np.linspace(real_x.min(), real_x.max(), len(all_sim[0]))  # Common x-grid over [0, 360)

    # Interpolate all CDF curves onto x_common
    y_interp = np.empty((num_sim, len(x_common)))

    for i in range(num_sim):
        x = all_sim[i]
        y = all_sim_cdf[i]

        # Sort x just in case
        sort_idx = np.argsort(x)
        x = x[sort_idx]
        y = y[sort_idx]

        # Interpolate onto common grid
        f_interp = interp1d(x, y, kind='nearest', bounds_error=False, fill_value=(0.0, 1.0))
        y_interp[i] = f_interp(x_common)

    return y_interp, x_common

def fast_bin_spectrum(wavelength, flux, bin_width):
    """
    Bin spectrum using fixed wavelength bin width (in μm).
    Much faster than manual looping.
    """
    min_wave = np.min(wavelength)
    max_wave = np.max(wavelength)
    bins = np.arange(min_wave, max_wave + bin_width, bin_width)

    # Compute mean flux in each bin
    binned_flux, _, _ = binned_statistic(wavelength, flux, statistic='mean', bins=bins)

    # Compute bin centers
    bin_centers = 0.5 * (bins[1:] + bins[:-1])

    # Filter out empty bins (NaNs)
    valid = ~np.isnan(binned_flux)
    return bin_centers[valid], binned_flux[valid]
def plot_sims(title, lon_360, lon_360_cdf, all_lon_sim, all_lon_sim_cdf, pvals_lon, lat, lat_cdf_vals, all_lat_sim, all_lat_sim_cdf, pvals_lat):
    # number of x's per bin
    n = 5
    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(16, 16), dpi=300)

    # Latitude CDF
    axes[0,0].plot(lat, lat_cdf_vals, marker='.', linestyle='-', color='#0072B2', label='Observed')
    # interpolate the simulated and observations
    y_interp_lat_sim, x_lat_sim = interp(all_lat_sim, all_lat_sim_cdf, lat)
    y_interp_lat_obs, x_lat_obs = interp([lat], [lat_cdf_vals], lat)
    lower_percentile_lat = np.percentile(y_interp_lat_sim, 5, axis=0)
    upper_percentile_lat = np.percentile(y_interp_lat_sim, 95, axis=0)
    median_lat = np.percentile(y_interp_lat_sim, 50, axis=0)

    # bin the sims and obs (since inconsistant slopes from point to point)
    binned_x_median_lat, binned_y_median_lat = fast_bin_spectrum(x_lat_sim, median_lat, n)
    binned_x_lower_lat, binned_y_lower_lat = fast_bin_spectrum(x_lat_sim, lower_percentile_lat, n)
    binned_x_upper_lat, binned_y_upper_lat = fast_bin_spectrum(x_lat_sim, upper_percentile_lat, n)
    binned_x_lat, binned_y_lat = fast_bin_spectrum(x_lat_obs, y_interp_lat_obs[0], n)
    d_lower_lat = np.gradient(binned_y_lower_lat)
    d_upper_lat = np.gradient(binned_y_upper_lat)
    d_median_lat = np.gradient(binned_y_median_lat)
    dy_lat = np.gradient(binned_y_lat)

    axes[0,0].plot(binned_x_median_lat, binned_y_median_lat, marker='.', linestyle='-', color='#CE619D', label='Simulated median (binned)')
    axes[0,0].fill_between(x_lat_sim, lower_percentile_lat, upper_percentile_lat, color='#CC79A7', alpha=.5, label='Simulated (middle 90%)')
    axes[0,0].set_xlabel('Latitude (degrees)', fontsize=17)
    axes[0,0].set_ylabel('CDF', fontsize=17)
    axes[0,0].set_title('CDF of Latitude: ' + title + '\np-value = ' + str(np.mean(pvals_lat)), fontsize=18)
    axes[0,0].tick_params(axis='both', labelsize=15)
    axes[0,0].grid()
    axes[0,0].legend(fontsize=17)
    
    # Latitude derivative
    axes[1,0].fill_between(binned_x_lower_lat, d_lower_lat, d_upper_lat, color='#CC79A7', alpha=.5, label='Simulated (binned middle 90%)')
    axes[1,0].plot(binned_x_median_lat, d_median_lat, marker='.', linestyle='-', color='#CC79A7', label='Simulated median(binned)')
    axes[1,0].plot(binned_x_lat, dy_lat, marker='.', linestyle='-', color='#0072B2', label='Observed (binned)')
    axes[1,0].set_xlabel('Latitude (degrees)', fontsize=17)
    axes[1,0].set_ylabel('CDF Slope', fontsize=17)
    axes[1,0].set_title('CDF Slope of Latitude: ' + title, fontsize=18)
    axes[1,0].tick_params(axis='both', labelsize=15)
    axes[1,0].grid()
    axes[1,0].legend(fontsize=17)

    # Longitude CDF
    axes[0, 1].plot(lon_360, lon_360_cdf, marker='.', linestyle='-', color='#0072B2', label='Observed')
    # interpolate the simulated and observations
    y_interp_lon_sim, x_lon_sim = interp(all_lon_sim, all_lon_sim_cdf, lon_360)
    y_interp_lon_obs, x_lon_obs = interp([lon_360], [lon_360_cdf], lon_360)
    lower_percentile_lon = np.percentile(y_interp_lon_sim, 5, axis=0)
    upper_percentile_lon = np.percentile(y_interp_lon_sim, 95, axis=0)
    median_lon = np.percentile(y_interp_lon_sim, 50, axis=0)

    # bin the sims and obs (since inconsistant slopes from point to point)
    binned_x_median_lon, binned_y_median_lon = fast_bin_spectrum(x_lon_sim, median_lon, n)
    binned_x_lower_lon, binned_y_lower_lon = fast_bin_spectrum(x_lon_sim, lower_percentile_lon, n)
    binned_x_upper_lon, binned_y_upper_lon = fast_bin_spectrum(x_lon_sim, upper_percentile_lon, n)
    binned_x_lon, binned_y_lon = fast_bin_spectrum(x_lon_obs, y_interp_lon_obs[0], n)
    d_median_lon = np.gradient(binned_y_median_lon)
    d_lower_lon = np.gradient(binned_y_lower_lon)
    d_upper_lon = np.gradient(binned_y_upper_lon)
    dy_lon = np.gradient(binned_y_lon)

    axes[0,1].plot(binned_x_median_lon, binned_y_median_lon, marker='.', linestyle='-', color='#CE619D', label='Simulated median (binned)')
    axes[0, 1].fill_between(x_lon_sim, lower_percentile_lon, upper_percentile_lon, color='#CC79A7', alpha=.5, label='Simulated (middle 90%)')
    axes[0, 1].set_xlabel('Longitude (degrees)', fontsize=17)
    axes[0, 1].set_ylabel('CDF', fontsize=17)
    axes[0, 1].set_title('CDF of Longitude: ' + title + '\np-value = '+ str(np.mean(pvals_lon)), fontsize=18)
    axes[0, 1].tick_params(axis='both', labelsize=15)
    axes[0, 1].grid()
    axes[0, 1].legend(fontsize=17)
    
    

    #Longitude derivative
    axes[1,1].fill_between(binned_x_upper_lon, d_lower_lon, d_upper_lon, color='#CC79A7', alpha =.5, label='Simulated (binned middle 90%)')
    axes[1,1].plot(binned_x_median_lon, d_median_lon, marker='.', linestyle='-', color='#CC79A7', label='Simulated median(binned)')
    axes[1,1].plot(binned_x_lon, dy_lon, marker='.', linestyle='-', color='#0072B2', label='Observed (binned)')
    # axes[1,1].set_yscale('log')
    axes[1,1].set_xlabel('Longitude (degrees)', fontsize=17)
    axes[1,1].set_ylabel('CDF Slope', fontsize=17)
    axes[1,1].set_title('CDF Slope of Longitude: ' + title, fontsize=18)
    axes[1,1].tick_params(axis='both', labelsize=15)
    axes[1,1].grid()
    axes[1,1].legend(fontsize=17)

    plt.tight_layout()
    plt.show()


def plot_cdf(boundary, title, stereo=False, just_stereo=False):
    """
    Plots the CDFs for the given boundary and simulates data.

    Parameters
    ----------
    boundary : str
        The boundary to use for the CDF plot, typically 'goes-w' or 'goes-e'.
    title : str
        The title for the plot, typically indicating the boundary and whether stereo mode is used.
    stereo : bool, optional
        Whether to use stereo mode (default is False).
    just_stereo : bool, optional
        If True, only samples within the stereo region will be used (default is False).
    """
    if boundary == 'goes-w':
        poly = poly_west
    elif boundary == 'goes-e':
        poly = poly_east
    elif boundary == 'goes':
        poly = fu.get_boundary('goes', collection=True, intersection=False, crs=None)
        
    else:
        raise ValueError("Unsupported boundary. Use 'goes-w', 'goes-e', or 'goes'.")
    
    bdf = BolideDataFrame(source='glm')
    # Filter out dates before July 1, 2019
    bdf_boundary = bdf.filter_boundary(boundary=boundary)
    bdf_boundary = bdf_boundary[bdf_boundary['datetime'] >= pd.Timestamp('2019-07-01', tz='UTC')]
    
    sdf = ShowerDataFrame()
    bdf_final = make_bdf_showers(bdf_boundary, sdf, 41)
            

    lon, lat = create_lonlat(bdf_final)

    if not stereo:
        # Filter out latitudes and longitudes that are in the stereo region
        accepted_lats = []
        accepted_lons = []

        for lat, lon in zip(lat, lon):
            if not in_stereo(lon, lat):
                accepted_lats.append(lat)
                accepted_lons.append(lon)
    elif just_stereo:
        # Filter out latitudes and longitudes that are not in the stereo region
        accepted_lats = []
        accepted_lons = []

        for lat, lon in zip(lat, lon):
            if in_stereo(lon, lat):
                accepted_lats.append(lat)
                accepted_lons.append(lon)
    else:
        # Use all latitudes and longitudes in the stereo region
        accepted_lats = lat
        accepted_lons = lon


    lat, lat_cdf_vals = cdf(accepted_lats)
    len_bdf = len(accepted_lats)
    # Prepare simulated latitudes/longitudes in the stereo region
    lon_samples, lat_samples = montecarlo(len_bdf, poly=poly, stereo=stereo, just_stereo=just_stereo)

    lon_sim, lon_sim_cdf = cdf(lon_samples)
    lat_sim, lat_sim_cdf = cdf(lat_samples)

    # Convert longitudes to [0, 360) for CDF
    lon_360 = wrap_lon(accepted_lons)
    sim_lon_360 = wrap_lon(lon_sim)
    lon_360, lon_360_cdf = cdf(lon_360)
    sim_lon_360, sim_lon_360_cdf = cdf(sim_lon_360)

    # Calculate the KS test
    cdf_lat = interp1d(lat, lat_cdf_vals, bounds_error=False, fill_value=(0.0, 1.0))
    cdf_lon = interp1d(lon_360, lon_360_cdf, bounds_error=False, fill_value=(0.0, 1.0))
    cdf_lat_sim = interp1d(lat_sim, lat_sim_cdf, bounds_error=False, fill_value=(0.0, 1.0))
    cdf_lon_sim = interp1d(sim_lon_360, sim_lon_360_cdf, bounds_error=False, fill_value=(0.0, 1.0))
    # ks_lat = scipy.stats.ks_1samp(lat, cdf_lat)
    ks_lat = scipy.stats.ks_2samp(accepted_lats, lat_sim)
    pvalue_lat = ks_lat[1]
    # ks_lon = scipy.stats.ks_1samp(lon_360, cdf_lon)
    ks_lon = scipy.stats.ks_2samp(lon_360, sim_lon_360)
    pvalue_lon = ks_lon[1]

    # indices = np.setdiff1d(np.arange(len(lat)), np.unique(lat, return_index=True)[1])
    # print(lat[indices])
    # y_lat = PchipInterpolator(lat, lat_cdf_vals, axis=0)
    # y_lat_sim = PchipInterpolator(lat_sim, lat_sim_cdf, axis=0)
    # y_lon_360 = PchipInterpolator(lon_360, lon_360_cdf, axis=0)
    # y_sim_lon_360 = PchipInterpolator(sim_lon_360, sim_lon_360_cdf, axis=0)



    dlat = np.gradient(lat_cdf_vals, lat)
    dlat_sim = np.gradient(lat_sim_cdf, lat_sim)
    dlon = np.gradient(lon_360_cdf, lon_360)
    dlon_sim = np.gradient(sim_lon_360_cdf, sim_lon_360)

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(16, 16), dpi=300)

    # Latitude CDF
    axes[0,0].plot(lat, lat_cdf_vals, marker='.', linestyle='-', color='#0072B2', label='Observed')
    axes[0,0].plot(lat_sim, lat_sim_cdf, marker='.', linestyle='-', color='#CC79A7', label='Simulated')
    axes[0,0].set_xlabel('Latitude (degrees)', fontsize=17)
    axes[0,0].set_ylabel('CDF', fontsize=17)
    axes[0,0].set_title('CDF of Latitude: ' + title + '\np-value = ' + str(pvalue_lat), fontsize=18)
    axes[0,0].tick_params(axis='both', labelsize=15)
    axes[0,0].grid()
    axes[0,0].legend(fontsize=17)

    # Longitude CDF
    axes[0, 1].plot(lon_360, lon_360_cdf, marker='.', linestyle='-', color='#0072B2', label='Observed')
    axes[0, 1].plot(sim_lon_360, sim_lon_360_cdf, marker='.', linestyle='-', color='#CC79A7', label='Simulated')
    axes[0, 1].set_xlabel('Longitude (degrees)', fontsize=17)
    axes[0, 1].set_ylabel('CDF', fontsize=17)
    axes[0, 1].set_title('CDF of Longitude: ' + title + '\np-value = '+ str(pvalue_lon), fontsize=18)
    axes[0, 1].tick_params(axis='both', labelsize=15)
    axes[0, 1].grid()
    axes[0, 1].legend(fontsize=17)

    # Latitude derivative
    axes[1,0].plot(lat, dlat, marker='.', linestyle='-', color='#0072B2', label='Observed')
    axes[1,0].plot(lat_sim, dlat_sim, marker='.', linestyle='-', color='#CC79A7', label='Simulated')
    axes[1,0].set_yscale('log')
    axes[1,0].set_xlabel('Latitude (degrees)', fontsize=17)
    axes[1,0].set_ylabel('CDF Slope', fontsize=17)
    axes[1,0].set_title('CDF Slope of Latitude: ' + title, fontsize=18)
    axes[1,0].tick_params(axis='both', labelsize=15)
    axes[1,0].grid()
    axes[1,0].legend(fontsize=17)

    #Longitude derivative
    axes[1,1].plot(lon_360, dlon, marker='.', linestyle='-', color='#0072B2', label='Observed')
    axes[1,1].plot(sim_lon_360, dlon_sim, marker='.', linestyle='-', color='#CC79A7', label='Simulated')
    axes[1,1].set_yscale('log')
    axes[1,1].set_xlabel('Longitude (degrees)', fontsize=17)
    axes[1,1].set_ylabel('CDF Slope', fontsize=17)
    axes[1,1].set_title('CDF Slope of Longitude: ' + title, fontsize=18)
    axes[1,1].tick_params(axis='both', labelsize=15)
    axes[1,1].grid()
    axes[1,1].legend(fontsize=17)

    plt.tight_layout()
    plt.show()
    
    # lat_cdf(lat, lat_cdf_vals, lat_sim, lat_sim_cdf, title=title)

    

    # Plot longitude CDFs
    # lon_cdf(lon_360, lon_360_cdf, sim_lon_360, sim_lon_360_cdf, title=title)