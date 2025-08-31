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
from sunpy.coordinates import sun  # para pegar B0 (tilt) na data do mapa

import warnings
from sunpy.util.exceptions import SunpyUserWarning
warnings.filterwarnings("ignore", category=SunpyUserWarning, module="sunpy.coordinates.frames")


def main(time_bins, flaws=None, outdir=Path(".")):
    """
    Function that receives as a parameter the time series to be analyzed.
    Returns Day, Latitudes, Areas, Number_of_Sunspot_Groups and saves CSVs.
    """
    if flaws is None:
        flaws = Time(['1998-11-16', '1998-11-17', '1998-11-18', '1998-11-19', '1998-11-20', '1999-03-19', '1999-04-15',
                      '2001-03-10', '2001-05-12', '2002-01-26', '2002-05-13', '2002-06-14', '2002-10-04', '2002-08-26',
                      '2003-03-09', '2004-03-09', '2004-05-07', '2004-07-17', '2005-04-05', '2005-06-30', '2005-07-24', 
                      '2006-04-01', '2006-04-15', '2006-04-01', '2007-04-30', '2007-05-16', '2008-06-17', '2009-11-12',
                      '2008-08-03', '2010-03-08', '2010-03-24', '2010-03-25', '2010-03-26', '2010-03-27', '2010-03-28', 
                      '2010-08-05', '2010-09-28', '2010-09-29', '2010-10-26', '2010-10-27', '2010-10-28', '2010-12-19',
                      '2010-12-20', '2010-12-21', '2010-12-22', '2010-12-23', '2010-12-24', '2010-12-25', '2010-12-26',
                      '2010-12-27', '2010-12-28', '2010-12-29', '2010-12-30', '2010-12-31', '2011-01-01', '2011-01-02',
                      '2011-01-03', '2011-01-04', '2011-01-05', '2011-01-06', '2011-01-07', '2011-01-08', '2011-01-09', 
                      '2011-01-10', '2011-01-11', '2011-01-12', '2011-01-13', '2011-01-14', '2011-01-15', '2011-01-16', 
                      '2011-11-23'])
        
        flaws_dates = set([s[:10] for s in Time(flaws).isot])


    Day = []   # time objects from maps
    Latitudes = [] # latitudes of the sunspots
    Longitudes = []
    Areas = [] # areas of the spots
    Number_of_Sunspot_Groups = []  # number of the spots groups per day

    # The original notebook accessed bins via time[i][0]; here we keep it compatible:
    # Build a list of Time objects representing each bin start.
    # BinnedTimeSeries supports iteration over 'time_bin_start' column.
    if hasattr(time_bins, "time_bin_start"):
        time_list = list(time_bins.time_bin_start)
    else:
        time_list = list(time_bins)

    for i in range(len(time_list)):
        day = Time(time_list[i])
        logging.info("Processing %s", day.isot)

        try:
            day_date = day.isot[:10]  # só a data
            if day_date in flaws_dates:

                Day.append(day.value)
                Latitudes.append(np.nan)
                Longitudes.append(np.nan)
                Areas.append(np.nan)
                Number_of_Sunspot_Groups.append(0)
                continue
            else:
                map_file =  get_map(day) # Create the coordinate map
                tophat_map = top_hat(day, map_file) # Apply the transform
                erode_map = erode(day, tophat_map) # Apply erosion to the transformed image
                sun_radius = get_sun_radius(map_file) # sun's radius in pixels
                contours = get_contours(tophat_map) # contours to sunspots
                group_contours = get_contours(erode_map) # contours to sunspots groups

                # --- novo: processa e conta quantos spots "aceitos" tivemos
                latitudes, longitudes, Am = get_coord_and_area(contours, tophat_map, sun_radius)

                accepted = 0
                for lat, lon, area in zip(latitudes, longitudes, Am):
                    Day.append(map_file.date.value)
                    Latitudes.append(lat)
                    Longitudes.append(lon)
                    Areas.append(area)
                    accepted += 1

                # se não aceitamos nenhum spot (zero contornos OU todos descartados pelo μ), registre NaNs para o dia
                if accepted == 0:
                    Day.append(map_file.date.value)
                    Latitudes.append(np.nan)
                    Longitudes.append(np.nan)
                    Areas.append(np.nan)


                # Note: original code appended len(contours); that likely counted spots not groups.
                # Here we keep minimal change but fix to use group_contours for groups.
                if len(group_contours) == 0:
                    Number_of_Sunspot_Groups.append(0)
                else:
                    Number_of_Sunspot_Groups.append(len(group_contours))

        except Exception as e:
            logging.exception("Failed on %s", day.isot)
            Day.append(Time(day))
            Latitudes.append(np.nan)
            Longitudes.append(np.nan)
            Areas.append(np.nan)
            Number_of_Sunspot_Groups.append(0)
            
    # Convert to Time objects
    lista = [Time(i) for i in Day]
    Day = lista

    # ---------- CSV 1: manchas individuais ----------
    # Data (ISO), Latitude e Área
    df1 = pd.DataFrame({
        "Date": [t.isot for t in Day],   # coluna de data explícita
        "Lat": Latitudes,
        "Lon": Longitudes,
        "Area": Areas
    })
    out1 = Path(outdir) / "Sunspots.csv"
    df1.to_csv(out1, index=False)

    # ---------- CSV 2: grupos por dia/bin ----------
    # Garante vetor de datas dos bins (ISO)
    try:
        tstarts = [Time(t).isot for t in getattr(time_bins, "time_bin_start")]
    except Exception:
        # fallback: se o objeto não tiver o atributo; mantém comprimento correto
        tstarts = [str(i) for i in range(len(Number_of_Sunspot_Groups))]

    df2 = pd.DataFrame({
        "Date": tstarts,                             # coluna de data explícita
        "Number_of_Sunspot_Groups": Number_of_Sunspot_Groups
    })
    out2 = Path(outdir) / "Groups.csv"
    df2.to_csv(out2, index=False)

    # ---------- Metadados de execução ----------
    start_input = tstarts[0] if len(tstarts) else None
    meta = {
        "script": "sunspots_pipeline.py",
        "inputs": {
            "start": start_input,
            "n_bins": len(df2),
            "bin_size": "1 d"
        },
        "outputs": [str(out1), str(out2)],
        "notes": "Auto-generated by FAIR-style script; dependent on hvpy/Helioviewer API availability."
    }
    (Path(outdir) / "run_metadata.json").write_text(json.dumps(meta, indent=2))

    return Day, Latitudes, Longitudes, Areas, Number_of_Sunspot_Groups


def get_map(day):
    if day < Time('2010-12-06'):  
        s_id = hvpy.DataSource.MDI_INT.value  # before 2010-12-06, use MDI instrument
    else:
        s_id = hvpy.DataSource.HMI_INT.value  # after 2010-12-06, use HMI instrument

    hmi_file = hvpy.save_file(hvpy.getJP2Image(day.datetime, s_id),
                              "T.JPEG2000", overwrite=True)  # get JP2
    map_file = Map(hmi_file)  # Create coordinate map from jp2 metadata using Sunpy

    # --- adiciona borda preta de 24px em todos os lados ---
    padded_data = cv.copyMakeBorder(map_file.data,
                                    top=26, bottom=26,
                                    left=26, right=26,
                                    borderType=cv.BORDER_CONSTANT,
                                    value=0)  # pixels pretos

    # cria novo Map com os mesmos metadados
    padded_map = Map(padded_data, map_file.meta)

    return padded_map



def top_hat(day, map_file):
    pixel_matrix = 255 - map_file.data # negative image

    if day < Time('2010-12-06'):
        pixel_matrix = cv.medianBlur(pixel_matrix, 5)
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(45,45))
    else:
        pixel_matrix = cv.medianBlur(pixel_matrix,15) 
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(135,135))
    
    tophat = cv.morphologyEx(pixel_matrix, cv.MORPH_TOPHAT, kernel)
    _, binary = cv.threshold(tophat,25,255,cv.THRESH_BINARY_INV)
    tophat_map = Map(binary, map_file.meta)
    return tophat_map


def get_sun_radius(map_file):
    hpc_coords = all_coordinates_from_map(map_file)
    mask = coordinate_is_on_solar_disk(hpc_coords)
    _ = measure.find_contours(mask, 1)
    labeled_mask, _ = ndimage.label(mask)
    regions = ndimage.find_objects(labeled_mask)

    radius = np.nan
    for r in regions:
        dy, dx = r
        radius = (dx.stop - dx.start)/2
    return radius


def erode(day, map_file):
    pixel_matrix = map_file.data 

    if day < Time('2010-12-06'):
        pixel_matrix = cv.medianBlur(pixel_matrix, 5)
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(45,45))
    else:
        pixel_matrix = cv.medianBlur(pixel_matrix,15) 
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(135,135))
    
    erode_img = cv.erode(pixel_matrix, kernel, iterations=1)
    erode_map = Map(erode_img, map_file.meta)
    return erode_map


def get_contours(tophat_map):
    threshold = 0
    binary_image = tophat_map.data == threshold
    contours, hierarchy = cv.findContours(binary_image.astype(np.uint8), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    return contours


def get_coord_and_area(contours, tophat_map, sun_radius, min_area_pix=16):
    """
    Para cada contorno válido:
      - calcula área em μHem pela soma pixel-a-pixel (μ = cos ρ)
      - registra lat/lon do centróide (HGS, graus)
    Só adiciona a mancha se a área foi calculada com sucesso (há pixels válidos).
    """
    latitudes, longitudes, Am_list = [], [], []

    for cnt in contours:
        As = cv.contourArea(cnt)
        if As <= min_area_pix:
            continue

        # 1) máscara do contorno (sem máscara de disco)
        mask = np.zeros(tophat_map.data.shape, dtype=np.uint8)
        cv.drawContours(mask, [cnt], contourIdx=-1, color=1, thickness=-1)
        mask = mask.astype(bool)

        ys, xs = np.nonzero(mask)
        if xs.size == 0:
            continue

        # 2) μ por pixel
        coord = tophat_map.pixel_to_world(xs * u.pix, ys * u.pix)
        cp = SkyCoord(coord.Tx, coord.Ty,
                      frame=frames.Helioprojective,
                      obstime=tophat_map.date, observer="earth")
        hgs = cp.transform_to(frames.HeliographicStonyhurst)

        lat_r = np.deg2rad(hgs.lat.to_value(u.deg))
        lon_r = np.deg2rad(hgs.lon.to_value(u.deg))
        B0 = np.deg2rad(sun.B0(tophat_map.date).value)

        mu = np.sin(B0) * np.sin(lat_r) + np.cos(B0) * np.cos(lat_r) * np.cos(lon_r)
        valid = np.isfinite(mu) & (mu > 0)
        if not np.any(valid):
            continue

        # 3) área μHem (soma 1/μ)
        sum_inv_mu = np.sum(1.0 / mu[valid])
        Am = (1e6 / (2 * np.pi * (sun_radius ** 2))) * sum_inv_mu

        # 4) agora sim, pegue lat/lon do centróide e registre TUDO junto
        M = cv.moments(cnt)
        if M['m00'] == 0:
            continue
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])

        coord_c = tophat_map.pixel_to_world(cx * u.pix, cy * u.pix)  # escalares!
        c = SkyCoord(coord_c.Tx, coord_c.Ty,
                     frame=frames.Helioprojective,
                     obstime=tophat_map.date, observer="earth")
        hgs_c = c.transform_to(frames.HeliographicStonyhurst)

        latitudes.append(hgs_c.lat.to_value(u.deg))
        longitudes.append(hgs_c.lon.to_value(u.deg))
        Am_list.append(float(Am))

    return latitudes, longitudes, Am_list




def build_bins(start: str, n_bins: int, bin_size_days: int = 1):
    """
    Create the BinnedTimeSeries given a start ISO string, number of bins, and bin size in days.
    """
    return BinnedTimeSeries(time_bin_start=start, time_bin_size=bin_size_days * u.d, n_bins=n_bins)


def parse_args():
    p = argparse.ArgumentParser(description="Detect sunspots and export CSVs (FAIR-style minimal script).")
    p.add_argument("--start", default="1998-01-01T12:00:00", help="Start time in ISO format (default: 1998-01-01T12:00:00)")
    p.add_argument("--n-bins", type=int, default=9709, help="Number of daily bins (default: 9709)")
    p.add_argument("--bin-size-days", type=int, default=1, help="Bin size in days (default: 1)")
    p.add_argument("--outdir", default=".", help="Output directory (default: current directory)")
    p.add_argument("--log-level", default="INFO", choices=["DEBUG","INFO","WARNING","ERROR","CRITICAL"], help="Logging level")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s %(levelname)s: %(message)s")
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    time = build_bins(args.start, args.n_bins, args.bin_size_days)

    # Run processing (uses original structure; only light CLI + logging added)
    Day, Latitudes, Longitudes, Areas, Number_of_Sunspot_Groups = main(time, outdir=outdir)

    print(f"Done. CSVs saved to: {outdir.resolve()}")
