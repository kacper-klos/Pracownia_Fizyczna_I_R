import scipy
import matplotlib.pyplot as plt
import numpy as np
import os

plt.rcParams.update({"font.size": 14})

IMAGE_PATH = "images"

DISTANCE_TIME = 120
DENSITY_TIME = 180
PUMP_TIME = 300

DISTANCE_ERROR = 0.05

VOLUME_ERROR = 0.01
TIME_VOLUME_ERROR = 1

SCYNTILLATOR_EFFICIENCY = 0.030625

distance_measurments = np.array([[36, 22, 12, 10 ,6], # N [bq]
                                 [1, 2, 3, 4, 5]]) # d[cm]

density_measurments = np.array([[132, 49, 91], # N_1 [bq]
                                [95, 35, 76],# N_2 [bq]
                                [2479.45, 2483.13, 2486.95],# V_1 [m^3]
                                [2483.13, 2486.95, 2490.87] # V_2 [m^3]
])
def LinearModel(params, x):
    m, c = params
    return m * x + c


def FitToLinearModel(x, y, x_err, y_err):
    linear = scipy.odr.Model(LinearModel)
    data = scipy.odr.RealData(x, y, sx=x_err, sy=y_err)
    initial_guess = [1.0, 0.0]

    odr = scipy.odr.ODR(data, linear, beta0=initial_guess)
    out = odr.run()
    return out.beta, out.sd_beta


def PlotLineFit(
    x,
    y,
    x_err,
    y_err,
    params,
    x_label,
    y_label,
    data_label,
    fit_label,
    title,
    line_color,
):
    plt.errorbar(x, y, xerr=x_err, yerr=y_err, fmt="o", label=data_label)
    x_fit = np.linspace(min(x), max(x), 200)
    y_fit = LinearModel(params, x_fit)
    plt.plot(x_fit, y_fit, f"-{line_color}", label=fit_label)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.savefig(os.path.join(IMAGE_PATH, f"{title}.png"), dpi=300, bbox_inches="tight")
    plt.clf()

def PlotDistanceMeasurments():
    plt.errorbar(distance_measurments[1], distance_measurments[1], xerr=DISTANCE_ERROR, yerr=np.sqrt(distance_measurments[1]), fmt="o", label="Punkty pomiarowe")
    plt.xlabel(r"$d$ [cm]")
    plt.ylabel(r"$N$ [Bq]")
    plt.legend()
    plt.savefig(os.path.join(IMAGE_PATH, f"{"distance"}.png"), dpi=300, bbox_inches="tight")
    plt.clf()

def DecayError(values):
    return np.sqrt(values)

def DistanceLogLog():
    x_log = np.log(distance_measurments[1])
    y_log = np.log(distance_measurments[0])
    x_err = DISTANCE_ERROR/distance_measurments[1]
    y_err = DecayError(distance_measurments[0])/distance_measurments[0]
    params_out, params_err = FitToLinearModel(x_log, y_log, x_err, y_err)
    print(f"params: {params_out}, error: {params_err}")
    PlotLineFit(x_log, y_log, x_err, y_err, params_out, r"$\log(N)$", r"$\log(d)$", "Punkty pomiarowe", "Dopasowanie liniowe", "distance_log", "r")

def Concentration(N_1, N_2, V_start, V_end, time):
    N_diff = N_1 - N_2
    N_err = DecayError(N_1+N_2)
    coefficient = 7.3/SCYNTILLATOR_EFFICIENCY*10e-5*time/(V_end-V_start)
    return coefficient*N_diff, coefficient*N_err

def ConcentrationMeasurments():
    concentration, concentration_err = Concentration(density_measurments[0], density_measurments[1], density_measurments[2], density_measurments[3], PUMP_TIME)
    print(f"concentration: {concentration}, errors: {concentration_err}")

DistanceLogLog()
PlotDistanceMeasurments()
# ConcentrationMeasurments()
