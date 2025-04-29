import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy
import os
import re

IMAGES_FOLDER = "images"
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
    plt.savefig(
        os.path.join(IMAGES_FOLDER, f"spektrum.png"), dpi=300, bbox_inches="tight"
    )
    plt.clf()


def FindSpectrumPeakWavelengths():
    wavelenght_ranges = [(450, 500), (500, 550), (600, 650)]
    peak_wavelengths = []
    peak_wavelengths_err = []
    for ran in wavelenght_ranges:
        filtered_spectrum_data = spectrum_data[
            (spectrum_data["wavelength"] >= ran[0])
            & (spectrum_data["wavelength"] <= ran[1])
        ]
        index = filtered_spectrum_data["intensity"].idxmax()
        peak_wavelengths.append(filtered_spectrum_data.loc[index, "wavelength"])

        max_intensity = filtered_spectrum_data.loc[index, "intensity"]
        filtered_error_spectrum_data = filtered_spectrum_data[filtered_spectrum_data["intensity"] >= 0.95*max_intensity]
        peak_wavelengths_err.append(max(np.abs(filtered_error_spectrum_data["wavelength"].to_numpy() - peak_wavelengths[-1])))
    return peak_wavelengths, peak_wavelengths_err


def PlotDetectorRun(run, num):
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
    plt.xlabel(r"$y \, \mathrm{[mm]}$")
    plt.ylabel(r"$I \, \mathrm{[\%]}$")
    plt.grid(True)
    plt.savefig(
        os.path.join(IMAGES_FOLDER, f"detektor{num}.png"), dpi=300, bbox_inches="tight"
    )
    plt.clf()


def FindRunWavelengths(runs):
    distance_ranges = [((30, 40), (40, 50), (50, 70)), ((30, 40), (40, 50), (50, 70))]
    distance_errors_intensity = [0.1, 0.05]
    peak_distance_full = []
    peak_distance_full_err = []

    for ind, run in enumerate(runs):
        peak_distance_run = []
        peak_distance_run_err = []
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
            
            filtered_detector_data_err = filtered_detector_data[
                    filtered_detector_data.iloc[:, DETECTOR_INTENSITY_COLUMN] >=
                (filtered_detector_data.iloc[pos, DETECTOR_INTENSITY_COLUMN] - distance_errors_intensity[ind])]
            peak_distance_run_err.append(max(np.abs(filtered_detector_data_err.iloc[:, DETECTOR_DISTANCE_COLUMN].to_numpy() - peak_distance_run[-1])))

        peak_distance_full.append(peak_distance_run)
        peak_distance_full_err.append(peak_distance_run_err)

    return peak_distance_full, peak_distance_full_err


def LinearModel(params, x):
    m, c = params
    return m * x + c


def FitToLinearModel(x, y, x_err, y_err):
    linear = scipy.odr.Model(LinearModel)
    data = scipy.odr.RealData(x, y, sx=x_err, sy=y_err)
    initial_guess = [1.0, 0.0]

    odr = scipy.odr.ODR(data, linear, beta0=initial_guess)
    out = odr.run()
    return out.beta, out.sd_beta, out.cov_beta*out.res_var


def PlotLineFit(
    x,
    y,
    x_err,
    y_err,
    params,
    x_label,
    y_label,
    title,
    line_color,
):
    plt.errorbar(x, y, xerr=x_err, yerr=y_err, fmt="o")
    x_fit = np.linspace(min(x), max(x), 200)
    y_fit = LinearModel(params, x_fit)
    plt.plot(x_fit, y_fit, f"-{line_color}")
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.savefig(os.path.join(IMAGES_FOLDER, f"{title}.png"), dpi=300, bbox_inches="tight")
    plt.clf()


def FitWavelengthToDistanceParams(a, b, a_err, b_err):
    A = 1 / a
    B = -b / a
    A_err = np.abs(a_err / (a**2))
    B_err = np.sqrt((b_err / a) ** 2 + (b * a_err / (a**2)) ** 2)
    return A, B, A_err, B_err


def WeightedAverage(data, data_err):
    inverse_err = np.sum(1 / data_err**2)
    top_val = np.sum(data / (data_err**2))
    val = top_val / inverse_err
    err = np.sqrt(1 / inverse_err)
    return val, err


def SpectrumAnalysis():
    PlotSpectrumData()
    print(FindSpectrumPeakWavelengths())


def DetectorAnalysis():
    grouped_detector_data = GroupDetectorData()
    usefull_detector_data = [grouped_detector_data[3], grouped_detector_data[4]]
    distance_peaks, distance_peaks_err = FindRunWavelengths(usefull_detector_data)
    for i, peak in enumerate(distance_peaks):
        print(f"peaks: {peak}, error: {distance_peaks_err[i]}")
    wavelength_peaks, wavelength_peaks_err = FindSpectrumPeakWavelengths()

    final_params_wavelength_up = 0
    final_params_wavelength_err = 0
    final_params_distance_up = 0
    final_params_distance_err = 0
    for i, distance in enumerate(distance_peaks):
        params, params_err, covar = FitToLinearModel(distance, wavelength_peaks, distance_peaks_err[i], wavelength_peaks_err)
        print(f"params for wavelength: {params}, error: {params_err}")
        print(f"covariance {covar}")
        final_params_wavelength_up += params/(params_err**2)
        final_params_wavelength_err += 1/(params_err**2)
        PlotLineFit(distance, wavelength_peaks, distance_peaks_err[i], wavelength_peaks_err, params, r"$y \, \mathrm{[mm]}$", r"$\lambda \, \mathrm{[nm]}$", f"line_fit_wavelength_{i}", "g")

        params, params_err, covar = FitToLinearModel(wavelength_peaks, distance, wavelength_peaks_err, distance_peaks_err[i])
        print(f"params for distance: {params}, error: {params_err}")
        print(f"covariance {covar}")
        final_params_distance_up += params/(params_err**2)
        final_params_distance_err += 1/(params_err**2)
        PlotLineFit(wavelength_peaks, distance, wavelength_peaks_err, distance_peaks_err[i], params, r"$\lambda \, \mathrm{[nm]}$", r"$y \, \mathrm{[mm]}$", f"line_fit_distance_{i}", "r")

    print(f"weighted params for wavelength: {final_params_wavelength_up/final_params_wavelength_err}, error: {np.sqrt(1/final_params_wavelength_err)}")
    print(f"weighted params for distance: {final_params_distance_up/final_params_distance_err}, error: {np.sqrt(1/final_params_distance_err)}")

# SpectrumAnalysis()
DetectorAnalysis()
