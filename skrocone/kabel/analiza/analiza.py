import scipy
import matplotlib.pyplot as plt
import numpy as np
import os

plt.rcParams.update({"font.size": 14})

IMAGE_PATH = "images"
NANO = 1e-9
PERCENT = 0.01

impedence = 75  # [ohm]
good_cable_distance_measurments = np.array(
    [
        [30, 60, 40, 80, 65, 130, 80, 160],  # d [m]
        [154.0, 304.0, 206.0, 406.0, 332.0, 662.0, 404.0, 810.0],
    ]
)  # t [nm]
good_cable_distance_measurments[1] *= NANO
bad_cable_distance_measurments = np.array(
    [
        [20, 40, 40, 80, 60, 120, 80, 160, 100, 200],  # d [m]
        [84.0, 164.0, 174.0, 346.0, 264.0, 524.0, 356.0, 708.0, 438.0, 872.0],
    ]
)  # t [nm]
bad_cable_distance_measurments[1] *= NANO
time_distance_err = 4 * NANO

good_cable_input_voltage = 2.840
good_cable_voltage_measurments = np.array(
    [
        [
            21.242,
            67.889,
            51.489,
            98.712,
            154.450,
            229.724,
            346.97,
            426.38,
            502.59,
            5.985,
        ],  # R [ohm]
        [-1.200, -0.080, -0.400, 0.320, 0.800, 1.080, 1.400, 1.600, 1.680, -1.920],
    ]
)  # U [V]
bad_cable_input_voltage = 2.960
bad_cable_voltage_measurments = np.array(
    [
        [
            5.949,
            21.741,
            50.467,
            73.712,
            99.180,
            154.913,
            228.870,
            324.13,
            423.34,
            502.51,
        ],  # R [ohm]
        [-1.760, -1.160, -0.440, -0.040, 0.280, 0.680, 0.960, 1.200, 1.480, 1.520],
    ]
)  # U [V]
cable_voltage_measurment_voltage_err = 0.04


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
    y_fit = Model(x_fit, *params)
    plt.plot(x_fit, y_fit, f"-{line_color}")
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.savefig(os.path.join(IMAGE_PATH, f"{title}.png"), dpi=300, bbox_inches="tight")
    plt.clf()


def CableDistanceAnalysis(measurments, title):
    params, params_err = scipy.optimize.curve_fit(
        LinearModel, measurments[0], measurments[1], sigma=time_distance_err
    )
    PlotLineFit(
        measurments[0], measurments[1], 0, time_distance_err, params, LinearModel, "", "", title, "r"
    )
    print(f"Parmas for distance: {params}, error: {np.sqrt(np.diag(params_err))}")


def ResistnaceError(data):
    first_range = 200
    second_range = 2000

    return_data = data
    return_data[data <= first_range] = (
        0.03 * PERCENT * data[data <= first_range] + 0.005 * PERCENT * first_range
    )
    return_data[data > first_range] = (
        0.02 * PERCENT * data[data > first_range] + 0.003 * PERCENT * second_range
    )

    return return_data


def AdjustVoltageDataToLinear(measurments, input_voltage):
    x = 1 / measurments[0]
    y = (input_voltage-measurments[1])/(input_voltage+measurments[1])
    x_err = np.abs(ResistnaceError(measurments[0]/(measurments[0]**2)))
    y_err = 2*cable_voltage_measurment_voltage_err*np.sqrt(((3*input_voltage+measurments[1])/((input_voltage + measurments[1])**2))**2)

    return x, y, 0, 0 

def VoltageMeasurmentModel(x, a, b):
    return b*(x-a)/(x+a)

def PlotRawData(measurments, title):
    plt.scatter(measurments[0], measurments[1])
    plt.savefig(os.path.join(IMAGE_PATH, f"{title}.png"), dpi=300, bbox_inches="tight")
    plt.clf()


def CableVoltageAnalysis(measurments, input_voltage, title):
    params, params_err = scipy.optimize.curve_fit(
        VoltageMeasurmentModel, measurments[0], measurments[1], sigma=cable_voltage_measurment_voltage_err 
    )
    PlotLineFit(
        measurments[0], measurments[1], 0, cable_voltage_measurment_voltage_err, params, VoltageMeasurmentModel, "", "", title, "r"
    )
    print(f"Parmas for distance: {params}, error: {np.sqrt(np.diag(params_err))}")

#CableDistanceAnalysis(good_cable_distance_measurments, "good_cable_distance")
#CableDistanceAnalysis(bad_cable_distance_measurments, "bad_cable_distance")
#PlotRawData(good_cable_voltage_measurments, "good_cable_voltage_raw")
#PlotRawData(bad_cable_voltage_measurments, "bad_cable_voltage_raw")
CableVoltageAnalysis(good_cable_voltage_measurments, good_cable_input_voltage, "good_cable_voltage")
CableVoltageAnalysis(bad_cable_voltage_measurments, bad_cable_input_voltage, "bad_cable_voltage")
