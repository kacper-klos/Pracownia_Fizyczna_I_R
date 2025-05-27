import scipy
import matplotlib.pyplot as plt
import numpy as np
import os

plt.rcParams.update({"font.size": 14})

IMAGE_PATH = "images"
ML_TO_M3 = 1e-6
CM_TO_M = 1e-2
ABSOLUTE_ZERO_TEMP = -273.15
KILO = 1e3
R = 8.314
SPHERE_VOLUME = np.pi*4/3*(2*2.54*CM_TO_M)**3

IZOHORIC_ERROR_PLOT_SCALE=1

syringe_data = [np.array([
        [60, 56, 52, 48, 44, 41, 37, 33, 29, 26], # V[ml]
        [99.6, 105.5, 110.4, 116.0, 121.1, 125.3, 130.5, 137.7, 144.8, 152.1], # P [kPa]
        ]), np.array([
        [26, 30, 34, 38, 42, 46, 50, 54, 58, 60], # V[ml]
        [97.6, 87.4, 78.8, 73.1, 68.0, 63.9, 60.7 , 57.9, 56.3, 55.6] # P [kPa]
        ])] 
syringe_temp = 23.1 - ABSOLUTE_ZERO_TEMP

izohoric_data = [np.array([
        [5.9, 12.7, 17.2, 23.6, 29.6, 34.7, 40.1, 46.3, 52.3, 58.5, 64.0, 71.1, 79.4, 84.8], # T [celcius]
        [95.1, 97.4, 98.9, 100.6, 102.7, 104.9, 106.5, 108.1, 108.9, 110.6, 113.8, 114.9, 119.4, 122.1] # P [kPa]
        ]), np.array([
        [7.9, 18.0, 27.0, 31.2, 37.1, 40.7, 47.5, 50.3, 53.9, 57.5, 60.9, 65.6, 70.0, 75.4, 81.6], # T [celcius]
        [87.6, 91.2, 94.1, 95.6, 98.3, 99.1, 100.7, 101.4, 102.6, 103.8, 104.6, 105.8, 106.9, 109.3, 110.2] # P [kPa]
        ])]

temperature_error = 0.5
pressure_error = 2
syringe_volume_error = 1.2


def LinearModel(x, a, b):
    return a * x + b

def PlotLineFit(
    x,
    y,
    x_err,
    y_err,
    params,
    Model,
    x_label,
    y_label,
    title,
    line_color,
    line_range_shift=[0,0]
):
    plt.errorbar(x, y, xerr=x_err, yerr=y_err, fmt="o", ms=7)
    if params is not None:
        ordered = np.sort(x);
        x_fit = np.linspace(ordered[line_range_shift[0]], ordered[len(ordered)-line_range_shift[1]-1], 200)
        y_fit = Model(x_fit, *params)
        plt.plot(x_fit, y_fit, f"-{line_color}")
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.savefig(os.path.join(IMAGE_PATH, f"{title}.png"), dpi=300, bbox_inches="tight")
    plt.clf()

def LineFitIzohoric(data):
    param, param_error = scipy.optimize.curve_fit(
        LinearModel, data[0], data[1], sigma=temperature_error
    )
    zero_temp = param[1]/param[0]
    zero_temp_error = zero_temp * np.sqrt(param_error[0][0]/(param[0]**2) + param_error[1][1]/(param[1]**2))
    moles = KILO*param[0]*SPHERE_VOLUME/R
    moles_error = np.sqrt(param_error[0][0])/param[0] * moles
    print(f"fitted params: {param}, error: {np.sqrt(np.diag(param_error))}")
    print(f"absolute zero temperature: {zero_temp}, error: {zero_temp_error}")
    print(f"moles: {moles}, error: {moles_error}")
    return param, param_error, zero_temp, zero_temp_error

def GassModel(pressure, pressure_error, volume, volume_error, temperature, temperature_error):
    fraction = pressure*volume/temperature
    fraction_error = fraction * np.sqrt((pressure_error/pressure)**2 + (volume_error/volume)**2 + (temperature_error/temperature)**2)
    return fraction, fraction_error

def LineFitIzotermic(data, gass_data, gass_data_error, range):
    param, param_error = scipy.optimize.curve_fit(
        LinearModel, data[1][range[0] : range[1]], gass_data[range[0] : range[1]], sigma=gass_data_error[range[0] : range[1]]
    )
    moles = param[1]/(R*KILO)
    moles_error = moles * np.sqrt(param_error[1][1])/param[1]
    print(f"fitted params: {param}, error: {np.sqrt(np.diag(param_error))}")
    print(f"moles: {moles}, error: {moles_error}")
    return param, param_error 

def SyringeAnalysis():
    linear_range = [[3, 10], [0,6]]
    plot_range = [[0,0], [0, 0]]
    for i, syringe_data_sample in enumerate(syringe_data):
        gass_data, gass_data_error = GassModel(syringe_data_sample[1], pressure_error, syringe_data_sample[0], syringe_volume_error, syringe_temp, temperature_error)
        param, param_error = LineFitIzotermic(syringe_data_sample, gass_data, gass_data_error, linear_range[i])
        PlotLineFit(syringe_data_sample[1], gass_data, pressure_error, gass_data_error, param, LinearModel, r"P $[kPa]$", r"$\frac{PV}{T}$ $[Pa \, L \, K^{-1}]$", f"izotermic_{i}", "r", plot_range[i])

def WeightedAverage(temp, temp_err):
    inverse_square = 1/temp_err**2
    inverse_sum = 1/(np.sum(inverse_square))
    temp_weighted = np.sum(inverse_square*temp)*inverse_sum
    temp_weighted_err = np.sqrt(inverse_sum)
    print(f"Weighted temperature: {temp_weighted}, error: {temp_weighted_err}")
    return temp_weighted, temp_weighted_err


def IzohoricAnalysis():
    temps = []
    temps_error = []
    for i, izohoric_data_sample in enumerate(izohoric_data):
        param, param_err, zero_temp, zero_temp_error = LineFitIzohoric(izohoric_data_sample)
        temps.append(zero_temp)
        temps_error.append(zero_temp_error)
        PlotLineFit(izohoric_data_sample[0], izohoric_data_sample[1], IZOHORIC_ERROR_PLOT_SCALE*temperature_error, IZOHORIC_ERROR_PLOT_SCALE*pressure_error, param, LinearModel, r"T $[C^\circ]$", r"P $[kPa]$", f"izohoric_{i}", "r")
    print()
    WeightedAverage(np.array(temps), np.array(temps_error))

# IzohoricAnalysis()
SyringeAnalysis()
