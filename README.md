# PyMoST – Python Morphological Sunspot Tracker

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17012872.svg)](https://doi.org/10.5281/zenodo.17012873)


**Harnessing Python to unveil the secrets of the Sun: PyMoST tracks and analyzes sunspots across solar cycles.**

We developed a tool called **Python Morphological Sunspot Tracker (PyMoST)**.
This code, written in Python, leverages the principles of **mathematical morphology** – a powerful image processing technique rooted in set theory.

The main goal of PyMoST is to identify and track sunspots, providing automated analysis of their physical evolution.

By employing PyMoST, researchers can gain valuable insights into the dynamic behavior of sunspots, which are crucial for understanding solar activity and its impact on space weather.
This tool not only enhances our ability to study sunspots with higher accuracy but also contributes to the broader field of solar physics by offering a robust method for continuous and detailed observation.

---

## Project structure

```
pymost/
├── src/
│   └── sunspots_pipeline.py
├── outputs/
│   ├── Sunspot.csv
│   ├── Groups.csv
│   └── run_metadata.json
├── sample/
│   ├── daily_area_total.txt
│   ├── GW_SS_Data.csv
│   ├── Kodaikanal_diario.txt
│   ├── SN_d_hem_V2.0.csv
│   └── Sample_Results_PyMoST.ipynb
├── CITATION.cff
├── LICENSE.txt
├── environment.yml
├── requirements.txt
└── README.md
```

---

## pymost/src/

* **sunspots_pipline.py** → Main structure of the code, containing the helper functions and the main function

---


## pymost/outputs

* **Sunspots.csv** → List of individual sunspots, with date in ISO format (column 1), latitude in degrees (column 2), longitude in degrees (column 3), and area in millionths of a solar hemisphere (column 4).
* **Groups.csv** → List of sunspot groups, with date in ISO format (column 1) and number of sunspot groups for the respective day (column 2).
* **run_metadata.json** → basic metadata of the run (start date, n\_bins, bin\_size, output paths).

---

## pymost/sample

* **daily_area_total.txt** → List of daily total sunspot areas from Greenwich (RGO and USAF / NOAA) for separate hemispheres and for the full disk, structured in six columns: year (column 1), month (column 2), day (column 3), total daily area (column 4), total daily area – northern hemisphere (column 5), and total daily area – southern hemisphere (column 6). Missing days within the dataset are indicated with area values of -1 in the daily sunspot area file.


* **GW_SS_Data.csv** → List of individual sunspots from Greenwich (RGO and USAF / NOAA), structured in six columns: year (column 1), month (column 2), day (column 3), observation time (column 4), latitude in degrees (column 5), and area (column 6). Note: the data have been limited by us starting from 1998.


* **Kodaikanal_diario.txt** → List of individual sunspots from Kodaikanal (KSO), with date in ISOT format (column 1), latitude in degrees (column 2), longitude in degrees (column 3), and area in millionths of a solar hemisphere (column 4). Note: the data have been limited by us starting from 1998.


* **SN_d_hem_V2.0.csv** → Daily sunspot numbers (ISN – SILSO/SIDC), total and by hemisphere. Contents:

Columns 1-3: Gregorian calendar date
Column 1: Year
Column 2: Month
Column 3: Day
Column 4: Date in fraction of year
Column 5: Daily total sunspot number.
Column 6: Daily North sunspot number.
Column 7: Daily South sunspot number.
Column 8: Standard deviation of raw daily total sunspot data
Column 9: Standard deviation of raw daily North sunspot data
Column 10: Standard deviation of raw daily South sunspot data
Column 11: Number of observations in daily total sunspot number
Column 12: Number of observations in daily North sunspot number (not determined yet: -1)
Column 13: Number of observations in daily South sunspot number (not determined yet: -1)
Column 14: Definitive/provisional marker. A blank indicates that the value is definitive. A '*' symbol indicates that the value is still provisional and is subject to a possible revision (Usually the last 3 to 6 months)

* **Sample_Results_PyMoST.ipynb** → Jupyter notebook for data processing and visualization

---

## Dependencies

The code is written in **Python version 3.13.5**.
The main libraries used are:

* **numpy (2.3.2)** – numerical operations
* **pandas (2.3.1)** – data tables and CSV manipulation
* **astropy (7.1.0)** – astronomical time series and coordinate handling
* **sunpy (7.0.1)** – solar data tools (maps, coordinates, etc.)
* **hvpy (1.1.0)** – Helioviewer Python API (for downloading JP2 images)
* **scipy (1.16.1)** – scientific functions and data processing
* **scikit-image (0.25.2)** – image processing (contours, etc.)
* **opencv-python-headless (4.12.0)** – morphological image processing
* **matplotlib (3.10.5)** (optional, for visualizations)

All dependencies are listed in `requirements.txt` and `environment.yml`.

---

## Installation

### Using Conda (recommended)

```bash
conda env create -f environment.yml
conda activate pymost
```

### Using pip + venv

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

---

## Usage

### Standard run

```bash
python src/sunspots_pipeline.py --start 1998-01-01T12:00:00 --n-bins 9709 --outdir outputs
```

Main parameters:

* `--start` → start date (ISO, e.g., 1998-01-01T12:00:00)
* `--n-bins` → number of time bins
* `--bin-size-days` → size of each bin in days (default = 1)
* `--outdir` → output directory (default = outputs/)

### Practical example (quick test)

To run a **short 30-day analysis starting from 2005-01-01**:

```bash
python src/sunspots_pipeline.py --start 2005-01-01T12:00:00 --n-bins 30 --outdir outputs/test_30d
```

This will create the following files in `outputs/test_30d/`:

* `Sunspot.csv` → individual sunspots (dates, latitudes, areas)
* `Groups.csv` → number of sunspot groups per day
* `run_metadata.json` → execution metadata

---


## License

This project is licensed under the terms of the
[Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)](https://creativecommons.org/licenses/by-nc/4.0/).

---

## Citation

If you use **PyMoST** in academic work, please cite it as described in the file [CITATION.cff](./CITATION.cff).
