#!/usr/bin/env python3


import argparse
import logging
from pathlib import Path
import json                

import numpy as np
import pandas as pd
import cv2 as cv
import hvpy
import astropy.units as u

from astropy.time import Time
from astropy.coordinates import SkyCoord
from astropy.timeseries import BinnedTimeSeries
from astropy.timeseries import TimeSeries

from scipy import ndimage
from skimage import measure

from sunpy.map import Map
from sunpy.coordinates import frames
from sunpy.map.maputils import all_coordinates_from_map, coordinate_is_on_solar_disk
from sunpy.coordinates import sun  

import warnings
from sunpy.util.exceptions import SunpyUserWarning
warnings.filterwarnings("ignore", category=SunpyUserWarning, module="sunpy.coordinates.frames")


def main(time_bins, flaws=None, outdir=Path(".")):
    
    """
    Main function that receives a time series (time bins) to be analyzed.  
    It returns lists with dates (Day), latitudes, longitudes, areas, number of sunspot groups,  
    and also saves the results as CSV files.  
    
    """

    if flaws is None:

        # List of known dates with defects (images with some capture flaw that interferes with proper data analysis)  

        flaws = Time(['1998-11-16', '1998-11-17', '1998-11-18', '1998-11-19', '1998-11-20', '1999-03-19', '1999-04-15', '2001-03-10', '2001-05-12', '2002-01-26', '2002-05-13', '2002-06-14', '2002-10-04', '2002-08-26',
                      '2003-03-09', '2004-03-09', '2004-05-07', '2004-07-17', '2005-04-05', '2005-06-30', '2005-07-24', '2006-04-01', '2006-04-15', '2006-04-01', '2007-04-30', '2007-05-16', '2008-06-17', '2009-11-12',
                      '2008-08-03', '2010-03-08', '2010-03-24', '2010-03-25', '2010-03-26', '2010-03-27', '2010-03-28', '2010-08-05', '2010-08-08', '2010-09-28', '2010-09-29', '2010-10-26', '2010-10-29', '2010-10-27',
                      '2010-10-28', '2010-10-30', '2010-11-15', '2010-11-16', '2010-11-17', '2010-12-03', '2010-12-04', '2010-12-05', '2010-12-19', '2010-12-20', '2010-12-21', '2010-12-22', '2010-12-23', '2010-12-24',
                      '2010-12-25', '2010-12-26', '2010-12-27', '2010-12-28', '2010-12-29', '2010-12-30', '2010-12-31', '2011-01-01', '2011-01-02', '2011-01-03', '2011-01-04', '2011-01-05', '2011-01-06', '2011-01-07', 
                      '2011-01-08', '2011-01-09', '2011-01-10', '2011-01-11', '2011-01-12', '2011-01-13', '2011-01-14', '2011-01-15', '2011-01-16', '2011-11-23', '2013-10-29'])  
                      
        flaws_dates = set([s[:10] for s in Time(flaws).isot]) # Create a set with the dates (YYYY-MM-DD only) to simplify checking  

    
    # ---------- Lists to store the analysis results ----------  

    Day = []   
    Latitudes = [] 
    Longitudes = [] 
    Areas = [] 
    Number_of_Sunspot_Groups = []  

    # Check if the object time_bins has the attribute "time_bin_start"
    # (in case it is a BinnedTimeSeries), otherwise use it directly as a list

    if hasattr(time_bins, "time_bin_start"):
        time_list = list(time_bins.time_bin_start)
    else:
        time_list = list(time_bins)

    # Loop over each time bin

    for i in range(len(time_list)):
        day = Time(time_list[i]) 
        logging.info("Processing %s", day.isot) 

        try:
            day_date = day.isot[:10]  # Take only the date in the format YYYY-MM-DD
            if day_date in flaws_dates:
                # If the date is in the defects list, add NaNs
                Day.append(day.value)
                Latitudes.append(np.nan)
                Longitudes.append(np.nan)
                Areas.append(np.nan)
                Number_of_Sunspot_Groups.append(0)
                continue
            else:
                # Otherwise, proceed with the analysis normally

                map_file =  get_map(day) # Create the coordinate map
                tophat_map = top_hat(day, map_file) # Apply the transform
                erode_map = erode(day, tophat_map) # Apply erosion to the transformed image
                sun_radius = get_sun_radius(map_file) # sun's radius in pixels
                contours = get_contours(tophat_map) # contours to sunspots
                group_contours = get_contours(erode_map) # contours to sunspots groups

                # Process sunspot coordinates and areas

                latitudes, longitudes, Am = get_coord_and_area(contours, tophat_map, sun_radius)

                accepted = 0 # contador de manchas aceitas
                for lat, lon, area in zip(latitudes, longitudes, Am):
                    Day.append(map_file.date.value)
                    Latitudes.append(lat)
                    Longitudes.append(lon)
                    Areas.append(area)
                    accepted += 1

                # If no spot is accepted (zero contours OR all discarded by μ), record NaNs for the day

                if accepted == 0: 
                    Day.append(map_file.date.value)
                    Latitudes.append(np.nan)
                    Longitudes.append(np.nan)
                    Areas.append(np.nan)


                # Count sunspot groups

                if len(group_contours) == 0:
                    Number_of_Sunspot_Groups.append(0)
                else:
                    Number_of_Sunspot_Groups.append(len(group_contours))

        except Exception as e: 
            # If an error occurs, log it and insert NaNs for the day
            logging.exception("Failed on %s", day.isot)
            Day.append(Time(day))
            Latitudes.append(np.nan)
            Longitudes.append(np.nan)
            Areas.append(np.nan)
            Number_of_Sunspot_Groups.append(0)
            
    # Convert to Time objects
    lista = [Time(i) for i in Day]
    Day = lista

    # ---------- CSV 1: individual sunspots ----------  

    # Date (ISO), Latitude and Area

    df1 = pd.DataFrame({
        "Date": [t.isot for t in Day],   # dates in ISO format
        "Lat": Latitudes,
        "Lon": Longitudes,
        "Area": Areas
    })
    out1 = Path(outdir) / "Sunspots.csv"
    df1.to_csv(out1, index=False)

    # ---------- CSV 2: groups per day/bin ----------
    
    try:
        # If it is a BinnedTimeSeries, take the dates directly
        tstarts = [Time(t).isot for t in getattr(time_bins, "time_bin_start")]
    except Exception:
        # Otherwise, create a list with numeric indices
        tstarts = [str(i) for i in range(len(Number_of_Sunspot_Groups))]

    df2 = pd.DataFrame({
        "Date": tstarts,                             
        "Number_of_Sunspot_Groups": Number_of_Sunspot_Groups
    })
    out2 = Path(outdir) / "Groups.csv"
    df2.to_csv(out2, index=False)

    # ---------- Execution metadata ----------

    start_input = tstarts[0] if len(tstarts) else None
    meta = {
        "script": "sunspots_pipeline.py",
        "inputs": {
            "start": start_input,
            "n_bins": len(df2),
            "bin_size": "1 d"
        },
        "outputs": [str(out1), str(out2)],
        "notes": "Auto-generated by PyMoST script; dependent on hvpy/Helioviewer API availability."
    }
    (Path(outdir) / "run_metadata.json").write_text(json.dumps(meta, indent=2))

    return Day, Latitudes, Longitudes, Areas, Number_of_Sunspot_Groups


def get_map(day):

    # Check the observation date: if it is before 2010-12-06,
    # use data from the MDI (SOHO) instrument; otherwise, use HMI (SDO).


    if day < Time('2010-12-06'):  
        s_id = hvpy.DataSource.MDI_INT.value  # before 2010-12-06, use MDI instrument
    else:
        s_id = hvpy.DataSource.HMI_INT.value  # after 2010-12-06, use HMI instrument

    # Download the JP2 image from Helioviewer for the chosen date and instrument.
    # "T.JPEG2000" is the temporary file name, overwrite=True forces overwriting.


    hmi_file = hvpy.save_file(hvpy.getJP2Image(day.datetime, s_id),
                              "T.JPEG2000", overwrite=True) 
    map_file = Map(hmi_file)   # Create a SunPy Map object from the JP2 file (contains data and metadata).

    # --- add a 24px black border on all sides, so that the Structuring Element has space for convolution ---
    
    padded_data = cv.copyMakeBorder(map_file.data,
                                    top=26, bottom=26,
                                    left=26, right=26,
                                    borderType=cv.BORDER_CONSTANT,
                                    value=0)  

    # New Map 
    padded_map = Map(padded_data, map_file.meta)

    return padded_map



def top_hat(day, map_file):
    pixel_matrix = 255 - map_file.data # Invert the image values (negative):
    
    # Preprocessing adjustment depends on the instrument:
    
    if day < Time('2010-12-06'): 

        # For MDI (SOHO) images:
        # apply a median filter with a 5×5 kernel to reduce noise.
        # Define an elliptical structuring element of 45×45 pixels.

        pixel_matrix = cv.medianBlur(pixel_matrix, 5)
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(45,45))
    else:
        # For HMI (SDO) images:
        # apply a stronger median filter with a 15×15 kernel.
        # Use a larger structuring element, 135×135 pixels.

        pixel_matrix = cv.medianBlur(pixel_matrix,15) 
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(135,135))

    # Apply the morphological Top-Hat transform.
    # This operation highlights small, dark structures in the inverted image.

    tophat = cv.morphologyEx(pixel_matrix, cv.MORPH_TOPHAT, kernel)
    
    # Apply thresholding: pixels below 25 become 255 (white),
    # the others become 0 (black), inverting the logic.

    _, binary = cv.threshold(tophat,25,255,cv.THRESH_BINARY_INV)
    
    # Create a new SunPy Map object with the resulting binary image,
    # keeping the metadata from the original map.

    tophat_map = Map(binary, map_file.meta)
    return tophat_map # Return the processed map (ready for sunspot detection).


def get_sun_radius(map_file):
    # Extract the helioprojective coordinates (HPC) of all pixels in the map.
    hpc_coords = all_coordinates_from_map(map_file)
    # Create a boolean mask indicating which pixels are inside the solar disk.
    mask = coordinate_is_on_solar_disk(hpc_coords)
    # Find the contours of the solar disk in the mask.
    _ = measure.find_contours(mask, 1)
    # Label the connected regions of the mask (here, the solar disk is one of them).
    labeled_mask, _ = ndimage.label(mask)
    # Get the boundaries (array slices) of the detected regions.
    regions = ndimage.find_objects(labeled_mask)

    radius = np.nan
    # For each region found (generally only 1: the solar disk):
    # row (y) and column (x) slices delimiting the disk
    # Calculate the radius as half of the region's width in pixels.
    for r in regions:
        dy, dx = r
        radius = (dx.stop - dx.start)/2
    return radius


def erode(day, map_file):
    pixel_matrix = map_file.data # Extract the data as a pixel matrix from the solar image

    if day < Time('2010-12-06'): # Set different parameters for the images
        pixel_matrix = cv.medianBlur(pixel_matrix, 5)
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(45,45))
    else:
        pixel_matrix = cv.medianBlur(pixel_matrix,15) 
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(135,135))
    
    # Apply the erosion operation: shrink white regions (detected spots),
    # merging nearby pixels into groups and eliminating small details.
    
    erode_img = cv.erode(pixel_matrix, kernel, iterations=1)
    erode_map = Map(erode_img, map_file.meta) # Create a new SunPy Map from the eroded image, preserving the metadata.
    
    return erode_map


def get_contours(tophat_map):
    threshold = 0 # Define the threshold value used to identify regions.

    # Create a binary image: pixels equal to 0 (black) are marked as True.
    # Note: in this case, it is capturing the dark regions of the transformed image.

    binary_image = tophat_map.data == threshold

    # Find the contours (edges) of the white regions in the binary image.
    # - binary_image converted to uint8 (values 0 or 1)
    # - cv.RETR_EXTERNAL → detects only external contours
    # - cv.CHAIN_APPROX_SIMPLE → simplifies the contour points

    contours, hierarchy = cv.findContours(binary_image.astype(np.uint8), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    # Return the list of contours found (each contour is an array of coordinates).

    return contours


def get_coord_and_area(contours, tophat_map, sun_radius, min_area_pix=16):
    """
    For each valid contour:
        - compute area in μHem by pixel-by-pixel summation (μ = cos ρ)
        - record centroid lat/lon (HGS, degrees)
    Only add the spot if the area was successfully calculated (there are valid pixels).
    """

    # Initialize output lists: latitudes, longitudes, and areas (μHem) per spot

    latitudes, longitudes, Am_list = [], [], []
    
    # Iterate through all detected contours

    for cnt in contours:
        # Contour area in pixels (OpenCV geometric approximation)
        As = cv.contourArea(cnt)
        # Discard very small contours (noise), below min_area_pix
        if As <= min_area_pix:
            continue

        # 1) Create a binary mask of the contour (fills the contour interior)
        #    Without applying a disk mask: considers the entire contour region in the image
        mask = np.zeros(tophat_map.data.shape, dtype=np.uint8)
        cv.drawContours(mask, [cnt], contourIdx=-1, color=1, thickness=-1)
        mask = mask.astype(bool)

        # Extract coordinates (row y, column x) of all pixels inside the contour        

        ys, xs = np.nonzero(mask)

        # If for some reason the mask is empty, skip

        if xs.size == 0:
            continue

        # 2) Convert each contour pixel to solar coordinates and compute μ
        #    pixel_to_world: (x,y) in pixels -> Helioprojective coordinates (Tx, Ty) with map WCS

        coord = tophat_map.pixel_to_world(xs * u.pix, ys * u.pix)
        
        # Build a SkyCoord in Helioprojective (HPC) with the map geometry/time

        cp = SkyCoord(coord.Tx, coord.Ty,
                      frame=frames.Helioprojective,
                      obstime=tophat_map.date, observer="earth")

        # Transform to Heliographic Stonyhurst (HGS), where we get solar lat/lon

        hgs = cp.transform_to(frames.HeliographicStonyhurst)

        # Convert latitude/longitude to radians

        lat_r = np.deg2rad(hgs.lat.to_value(u.deg))
        lon_r = np.deg2rad(hgs.lon.to_value(u.deg))

        # Solar axis tilt angle (B0) on the observation date, in radians

        B0 = np.deg2rad(sun.B0(tophat_map.date).value)

        # μ = cos(ρ) = sin(B0)*sin(lat) + cos(B0)*cos(lat)*cos(lon)
        # Projection value (cosine of the center-to-limb angle) for each pixel

        mu = np.sin(B0) * np.sin(lat_r) + np.cos(B0) * np.cos(lat_r) * np.cos(lon_r)

        # Valid pixels: finite μ and strictly positive (discard outside the disk/limb)

        valid = np.isfinite(mu) & (mu > 0)

        # If no valid pixel, skip this contour

        if not np.any(valid):
            continue

        # 3) Convert area: sum of contributions 1/μ per valid pixel
        #    Formula: Am = (10^6 / (2π R^2)) * Σ(1/μ_i), where R is the solar radius in pixels

        sum_inv_mu = np.sum(1.0 / mu[valid])
        Am = (1e6 / (2 * np.pi * (sun_radius ** 2))) * sum_inv_mu

        # 4) Compute the contour centroid to record a single lat/lon for the spot
        M = cv.moments(cnt)

        # If null moment (degenerate), skip

        if M['m00'] == 0:
            continue

        # Centroid coordinates (in pixels, integers)

        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])

        # Convert centroid to world (HPC) — here scalars (a single point)

        coord_c = tophat_map.pixel_to_world(cx * u.pix, cy * u.pix)  # escalares!
        c = SkyCoord(coord_c.Tx, coord_c.Ty,
                     frame=frames.Helioprojective,
                     obstime=tophat_map.date, observer="earth")
        
        # Transform centroid to HGS (solar latitude/longitude)

        hgs_c = c.transform_to(frames.HeliographicStonyhurst)

        # Add to output vectors: latitude (deg), longitude (deg), and area (μHem)

        latitudes.append(hgs_c.lat.to_value(u.deg))
        longitudes.append(hgs_c.lon.to_value(u.deg))
        Am_list.append(float(Am))

    return latitudes, longitudes, Am_list




def build_bins(start: str, n_bins: int, bin_size_days: int = 1):
    """
    Create a BinnedTimeSeries object with user-defined parameters.

    Parameters:
        - start: string in ISO format (e.g., "1998-01-01T18:00:00") → start date
        - n_bins: number of bins (time intervals) to be created
        - bin_size_days: size of each bin in days (default = 1 day)

    Returns:
        - An astropy BinnedTimeSeries object containing the configured bins.
    """

    # Create the BinnedTimeSeries from the provided arguments.
    # time_bin_start → start date
    # time_bin_size  → size of each bin (in days, converted to astropy units)
    # n_bins         → number of bins


    return BinnedTimeSeries(time_bin_start=start, time_bin_size=bin_size_days * u.d, n_bins=n_bins)


def parse_args():

    # Create an ArgumentParser object, which defines how the command line will be interpreted.
    # The description serves as documentation and appears when running the script with --help.

    p = argparse.ArgumentParser(description="Detect sunspots and export CSVs (FAIR-style minimal script).")


    # Argument --start: defines the start date of the analysis.
    # ISO format (YYYY-MM-DDTHH:MM:SS).
    # Default value: "1998-01-01T12:00:00".

    p.add_argument("--start", default="1998-01-01T12:00:00", help="Start time in ISO format (default: 1998-01-01T12:00:00)")

    # Argument --n-bins: number of time bins to generate (daily intervals).
    # Example: if n_bins=30 → analysis covers 30 days.
    # Type: integer. Default: 9709 (covers from 1998 to 2025).

    p.add_argument("--n-bins", type=int, default=9709, help="Number of daily bins (default: 9709)")

    # Argument --bin-size-days: size of each bin in days.
    # Type: integer. Default: 1 (daily bin).

    p.add_argument("--bin-size-days", type=int, default=1, help="Bin size in days (default: 1)")

    # Argument --outdir: directory where output files (CSVs/JSON) will be saved.
    # Default: current directory ".".

    p.add_argument("--outdir", default=".", help="Output directory (default: current directory)")

    # Argument --log-level: logging detail level.
    # Can be DEBUG, INFO, WARNING, ERROR, or CRITICAL.
    # Default: INFO.

    p.add_argument("--log-level", default="INFO", choices=["DEBUG","INFO","WARNING","ERROR","CRITICAL"], help="Logging level")

    # Parse the arguments provided by the user and return them as an object.

    return p.parse_args()


if __name__ == "__main__":

    # This block is only executed when the file is run directly
    # (e.g., "python sunspots_pipeline.py"), and not when imported as a module.

    # Read command-line arguments using the parse_args() function

    args = parse_args()

    # Configure the logging system (status/error messages).
    # The log level is defined by the --log-level argument.
    # The format includes timestamp, level, and message.

    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s %(levelname)s: %(message)s")

    # Define the output directory (convert string → Path).

    outdir = Path(args.outdir)

    # Create the output directory if it does not exist (including subfolders).

    outdir.mkdir(parents=True, exist_ok=True)

    # Build the time bins according to the provided parameters:
    # start date (--start), number of bins (--n-bins), and bin size (--bin-size-days).

    time = build_bins(args.start, args.n_bins, args.bin_size_days)

    # Run processing (uses original structure; only light CLI + logging added)
    Day, Latitudes, Longitudes, Areas, Number_of_Sunspot_Groups = main(time, outdir=outdir)

    # Print to the terminal a confirmation of execution and where the files were saved.

    print(f"Done. CSVs saved to: {outdir.resolve()}")
