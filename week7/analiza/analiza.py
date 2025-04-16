import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy
import os
import re

IMAGES_FOLDER = "imgaes"
ANALYSIS_FOLDER = "analiza"
SPECTRUM_FILE = "spektrum.csv"
DETECTOR_FILE = "detektor.csv"

spectrum_data = pd.read_csv(
    os.path.join(ANALYSIS_FOLDER, SPECTRUM_FILE), sep=";", decimal=","
)
detector_data = pd.read_csv(
    os.path.join(ANALYSIS_FOLDER, DETECTOR_FILE), sep=";", decimal=","
)
DETECTOR_DISTANCE_COLUMN = 1
DETECTOR_INTENSITY_COLUMN = 2


def GroupDetectorData():
    run_groups = {}
    for col in detector_data.columns:
        match = re.search(r"Run\s*#(\d+)", col)
        if match:
            run_number = match.group(1)
            run_groups.setdefault(run_number, []).append(col)

    list_of_dfs = [detector_data[run_groups[run]].copy() for run in sorted(run_groups)]
    return list_of_dfs


def PlotSpectrumData():
    wavelength_data = spectrum_data["wavelength"].to_numpy()
    intensity_data = spectrum_data["intensity"].to_numpy()

    plt.plot(
        wavelength_data,
        intensity_data,
        marker="o",
        linestyle="-",
        markersize=1,
        linewidth=1,
    )
    plt.xlabel(r"$\lambda \, \mathrm{[nm]}$")
    plt.ylabel(r"$I$")
    plt.grid(True)
    plt.show()


def FindSpectrumPeakWavelengths():
    wavelenght_ranges = [(450, 500), (500, 550), (600, 650)]
    peak_wavelengths = []
    for ran in wavelenght_ranges:
        filtered_spectrum_data = spectrum_data[
            (spectrum_data["wavelength"] >= ran[0])
            & (spectrum_data["wavelength"] <= ran[1])
        ]
        index = filtered_spectrum_data["intensity"].idxmax()
        peak_wavelengths.append(filtered_spectrum_data.loc[index, "wavelength"])
    return peak_wavelengths


def PlotDetectorRun(run):
    position_data = run.iloc[:, DETECTOR_DISTANCE_COLUMN].to_numpy()
    intensity_data = run.iloc[:, DETECTOR_INTENSITY_COLUMN].to_numpy()

    plt.plot(
        position_data,
        intensity_data,
        marker="o",
        linestyle="-",
        markersize=1,
        linewidth=1,
    )
    plt.xlabel(r"$s \, \mathrm{[mm]}$")
    plt.ylabel(r"$I \, \mathrm{[\%]}$")
    plt.grid(True)
    plt.show()


def FindRunWavelengths(runs):
    distance_ranges = [((30, 40), (40, 50), (50, 70)), ((30, 40), (40, 50), (50, 70))]
    peak_distance_full = []

    for ind, run in enumerate(runs):
        peak_distance_run = []
        for ran in distance_ranges[ind]:
            filtered_detector_data = run[
                (run.iloc[:, DETECTOR_DISTANCE_COLUMN] >= ran[0])
                & (run.iloc[:, DETECTOR_DISTANCE_COLUMN] <= ran[1])
            ]
            index = filtered_detector_data.iloc[:, DETECTOR_INTENSITY_COLUMN].idxmax()
            pos = filtered_detector_data.index.get_loc(index)
            peak_distance_run.append(
                filtered_detector_data.iloc[pos, DETECTOR_DISTANCE_COLUMN]
            )
        peak_distance_full.append(peak_distance_run)

    return peak_distance_full


def LinearFunction(x, a, b):
    return a * x + b


def FitDistanceToWavelength(distance, wavelength):
    params, params_err = scipy.optimize.curve_fit(LinearFunction, distance, wavelength)
    return params, params_err

def FitWavelengthToDistanceParams(a, b, a_err, b_err):
    A = 1/a
    B = -b/a
    A_err = np.abs(a_err/(a**2))
    B_err = np.sqrt((b_err/a)**2 + (b*a_err/(a**2))**2)
    return A, B, A_err, B_err

def WeightedAverage(data, data_err):
    inverse_err = np.sum(1/data_err**2)
    top_val = np.sum(data/(data_err**2))
    val = top_val/inverse_err
    err = np.sqrt(1/inverse_err)
    return val, err

def SpectrumAnalysis():
    PlotSpectrumData()
    print(FindSpectrumPeakWavelengths())

def DetectorAnalysis():
    grouped_detector_data = GroupDetectorData()
    usefull_detector_data = [grouped_detector_data[3], grouped_detector_data[4]]
    distance_peaks = FindRunWavelengths(usefull_detector_data)
    wavelength_peaks = FindSpectrumPeakWavelengths()

    for distance in distance_peaks:
        print(FitDistanceToWavelength(distance, wavelength_peaks))

    for i in usefull_detector_data:
        PlotDetectorRun(i)


# SpectrumAnalysis()
DetectorAnalysis()
