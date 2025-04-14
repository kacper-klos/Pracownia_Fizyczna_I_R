import scipy
import matplotlib.pyplot as plt
import numpy as np
import os

plt.rcParams.update({"font.size": 14})

IMAGE_PATH = "images"
CENTY = 0.01

RED_WAVELENGTH = 640e-9
red_wavelength_err = 12e-9
red_measurments = np.array(
    [
        [0.33, 0.44, 0.52, 0.60, 0.67, 0.74, 0.78, 0.84, 0.88, 0.92],
        [0.32, 0.44, 0.52, 0.60, 0.67, 0.73, 0.78, 0.84, 0.88, 0.92],
    ]
)
red_measurments *= CENTY
red_measurment_error = 0.01 * CENTY

GREEN_WAVELENGTH = 515e-9
green_wavelength_err = 25e-9
green_measurments = np.array(
    [
        [0.29, 0.39, 0.47, 0.54, 0.59, 0.65, 0.70, 0.75, 0.79, 0.84],
        [0.30, 0.40, 0.48, 0.54, 0.60, 0.66, 0.70, 0.76, 0.79, 0.84],
    ]
)
green_measurments *= CENTY
green_measurement_error = 0.01 * CENTY

BLUE_WAVELENGTH = 468e-9
blue_wavelength_err = 20e-9
blue_measurments = np.array(
    [
        [0.22, 0.34, 0.42, 0.49, 0.55, 0.61, 0.66, 0.71, 0.75, 0.79],
        [0.22, 0.34, 0.42, 0.49, 0.55, 0.61, 0.66, 0.71, 0.75, 0.79],
    ]
)
blue_measurments *= CENTY
blue_measurement_error = 0.01 * CENTY

measurment_error = np.ones(np.shape(red_measurments)[1])
measurment_error[0] = 2
measurment_error[2] = 2
measurment_error *= 0.01 * CENTY


def LinearModel(x, a, b):
    return a * x + b


def FitToLinearModel(x, y, y_err):
    param, param_cov = scipy.optimize.curve_fit(LinearModel, x, y, sigma=y_err)
    return param, param_cov


def PlotLineFit(
    x,
    y,
    y_err,
    params,
    x_label,
    y_label,
    data_label,
    fit_label,
    title,
    line_color,
):
    plt.errorbar(x, y, yerr=y_err, fmt="o", label=data_label)
    x_fit = np.linspace(min(x), max(x), 200)
    y_fit = LinearModel(x_fit, *params)
    plt.plot(x_fit, y_fit, f"-{line_color}", label=fit_label)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.savefig(os.path.join(IMAGE_PATH, f"{title}.png"), dpi=300, bbox_inches="tight")
    plt.clf()


def Radious(params, params_err, wavelength, wavelength_err):
    radious = 2 * params[0] / wavelength
    radious_err = radious * np.sqrt(
        params_err[0][0] / (params[0]) ** 2 + (wavelength_err / wavelength) ** 2
    )
    return radious, radious_err


def ColorFit(data, data_err, wavelength, wavelength_err, title, color):
    data_avg = (np.mean(data, axis=0) / 2) ** 2
    data_input = [i for i in range(1, data_avg.size + 1)]
    data_mean_err = data_avg * data_err
    param, param_err = FitToLinearModel(data_input, data_avg, data_mean_err)
    print(
        f"fited params for {title}: {param}, error: {[np.sqrt(param_err[0][0]), np.sqrt(param_err[1][1])]}"
    )
    PlotLineFit(
        data_input,
        data_avg,
        data_mean_err,
        param,
        r"\(k\)",
        r"\(r_k^2\)",
        "punkty pomiarowe",
        "dopasowanie liniowe",
        title,
        color,
    )
    radious, radious_err = Radious(param, param_err, wavelength, wavelength_err)
    print(f"radious for {title}: {radious}, error: {radious_err}")
    return radious, radious_err


def RadiousAnalysis(radious_and_err):
    final_radious_up = 0
    final_radious_err_inv = 0
    for i in radious_and_err:
        final_radious_err_inv += (1 / (i[1])) ** 2
        final_radious_up += i[0] / (i[1] ** 2)

    radious_final = final_radious_up / final_radious_err_inv
    radious_final_err = 1 / final_radious_err_inv
    print(f"final radious: {radious_final}, error: {radious_final_err}")


radious_and_err = []
radious_and_err.append(
    ColorFit(
        red_measurments,
        measurment_error,
        RED_WAVELENGTH,
        red_wavelength_err,
        "red",
        "r",
    )
)
radious_and_err.append(
    ColorFit(
        green_measurments,
        measurment_error,
        GREEN_WAVELENGTH,
        green_wavelength_err,
        "green",
        "g",
    )
)
radious_and_err.append(
    ColorFit(
        blue_measurments,
        measurment_error,
        BLUE_WAVELENGTH,
        blue_wavelength_err,
        "blue",
        "b",
    )
)

RadiousAnalysis(radious_and_err)
