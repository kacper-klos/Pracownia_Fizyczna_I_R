import scipy
import matplotlib.pyplot as plt
import numpy as np
import os

IMAGE_PATH = "images"
CENTY = 100

RED_WAVELENGTH = 620e-9
red_measurments = np.array([[0.33, 0.44, 0.52, 0.60, 0.67, 0.74, 0.78, 0.84, 0.88, 0.92],
                            [0.32, 0.44, 0.52, 0.60, 0.67, 0.73, 0.78, 0.84, 0.88, 0.92]])
red_measurments /= CENTY 
red_error = 0.01/CENTY

GREEN_WAVELENGTH = 515e-9
green_measurments = np.array([[0.29, 0.39, 0.47, 0.54, 0.59, 0.65, 0.70, 0.75, 0.79, 0.84],
                              [0.30, 0.40, 0.48, 0.54, 0.60, 0.66, 0.70, 0.76, 0.79, 0.84]])
green_measurments /= CENTY 
green_error = 0.01/CENTY

BLUE_WAVELENGTH = 465e-9
blue_measurments = np.array([[0.22, 0.34, 0.42, 0.49, 0.55, 0.61, 0.66, 0.71, 0.75, 0.79],
                             [0.22, 0.34, 0.42, 0.49, 0.55, 0.61, 0.66, 0.71, 0.75, 0.79]])
blue_measurments /= CENTY 
blue_error = 0.01/CENTY


def LinearModel(x, a, b):
    return a * x + b


def FitToLinearModel(x, y, y_err):
    param, param_cov = scipy.optimize.curve_fit(LinearModel, x, y, sigma=y_err)
    return param, param_cov


def PlotLineFit(
    x,
    y,
    x_err,
    params,
    x_label,
    y_label,
    data_label,
    fit_label,
    title,
    line_color,
):
    plt.errorbar(x, y, xerr=x_err, fmt="o", label=data_label)
    x_fit = np.linspace(min(x), max(x), 200)
    y_fit = LinearModel(x_fit, *params)
    plt.plot(x_fit, y_fit, f"-{line_color}", label=fit_label)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.savefig(os.path.join(IMAGE_PATH, f"{title}.png"), dpi=300, bbox_inches="tight")
    plt.clf()

def ColorFit(data, data_err, wavelength, title, color):
    data_avg = (np.mean(data, axis=0))**2
    data_input = [i for i in range(1, data_avg.size+1)]
    data_mean_err = data_err/(data.shape[0])
    param, param_err = FitToLinearModel(data_input, data_avg, data_mean_err)
    radious = 2*param[0]/wavelength
    PlotLineFit(data_input, data_avg, data_mean_err, param, "", "", "", "", title, color)
    print(radious)

ColorFit(red_measurments, red_error, RED_WAVELENGTH, "red", "r")
ColorFit(green_measurments, green_error, GREEN_WAVELENGTH, "green", "g")
ColorFit(blue_measurments, blue_error, BLUE_WAVELENGTH, "blue", "b")


