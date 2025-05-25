import scipy
import matplotlib.pyplot as plt
import numpy as np
import os

plt.rcParams.update({"font.size": 14})

IMAGE_PATH = "images"
ML_TO_M3 = 10e-3
ABSOLUTE_ZERO_TEMP = -273

siringe_data = [np.array([
        [60, 56, 52, 48, 44, 41, 37, 33, 29, 26], # V[ml]
        [99.6, 105.5, 110.4, 116.0, 121.1, 125.3, 127.5, 133.7, 144.8, 152.1], # P [kPa]
        ]), np.array([
        [26, 30, 34, 38, 42, 46, 50, 54, 58, 60], # V[ml]
        [99.6, 87.4, 78.8, 73.1, 68.0, 63.9, 60.7 , 58.9, 56.3, 55.6] # P [kPa]
        ])] 
siringe_temp = 23.1 + ABSOLUTE_ZERO_TEMP

izohoric_data = [np.array([
        [5.9, 12.7, 17.2, 23.6, 29.6, 34.7, 40.1, 46.3, 52.3, 58.5, 64.0, 71.1, 79.4, 84.8], # T [celcius]
        [95.1, 97.4, 98.9, 100.6, 102.7, 104.9, 106.5, 108.1, 108.9, 110.6, 111.8, 112.9, 113.7, 116.1] # P [kPa]
        ]), np.array([
        [7.9, 18.0, 27.0, 31.2, 37.1, 40.7, 47.5, 50.3, 53.9, 57.5, 60.9, 65.6, 70.0, 75.4, 81.6], # T [celcius]
        [87.6, 91.2, 94.1, 95.6, 98.3, 99.1, 100.7, 101.4, 102.6, 103.8, 104.6, 105.8, 106.9, 107.3, 110.2] # P [kPa]
        ])]

temperature_error = 0.01
volume_error = 0.01
pressure_error = 0.01


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
):
    plt.errorbar(x, y, xerr=x_err, yerr=y_err, fmt="o", ms=3)
    x_fit = np.linspace(min(x), max(x), 200)
    if params is not None:
        y_fit = Model(x_fit, *params)
        plt.plot(x_fit, y_fit, f"-{line_color}")
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.savefig(os.path.join(IMAGE_PATH, f"{title}.png"), dpi=300, bbox_inches="tight")
    plt.clf()

def LineFitIzohoric(data):
    param, param_error = scipy.optimize.curve_fit(
        LinearModel, data[0], data[1], sigma=temperature_error*data[0]
    )
    zero_temp = param[1]/param[0]
    zero_temp_error = zero_temp * np.sqrt(param_error[0][0]/(param[0]**2) + param_error[1][1]/(param[1]**2))
    print(f"fitted params: {param}, error: {param_error}")
    print(f"absolute zero temperature: {zero_temp}, error: {zero_temp_error}")
    return param, param_error, zero_temp, zero_temp_error

def GassModel(pressure, volume, temperature):
    return pressure*volume/temperature

def SingleSiringePlot(data):
    gass_data = GassModel(data[1], data[0], siringe_temp)

def SiringeAnalysis():


def IzohoricAnalysis():
    for i, izohoric_data_sample in enumerate(izohoric_data):
        param, param_err, zero_temp, zero_temp_error = LineFitIzohoric(izohoric_data_sample)
        PlotLineFit(izohoric_data_sample[0], izohoric_data_sample[1], temperature_error * izohoric_data_sample[0], pressure_error * izohoric_data_sample[1], param, LinearModel, "", "", f"izohoric_{i}", "r")

IzohoricAnalysis()


