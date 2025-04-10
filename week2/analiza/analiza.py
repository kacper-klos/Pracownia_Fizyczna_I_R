import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy

plt.rcParams.update({"font.size": 14})

ALPHA = 0.0045
CAPA = 144

R_0 = np.array([2.870, 2.888, 2.902])
R = np.array([9.802, 9.786, 9.796])

U_0_P = np.array(
    [
        0.11,
        0.21,
        0.31,
        0.41,
        0.51,
        0.61,
        0.71,
        0.81,
        0.91,
        1.01,
        1.11,
        1.21,
        1.31,
        1.41,
        1.51,
        1.76,
        2.01,
        2.25,
        2.5,
        3.01,
        3.51,
        4.51,
        5.51,
        7.50,
        9.51,
    ]
)
U_0_P_err = U_0_P * 0.0005 + 0.01
U_R_P = np.array(
    [
        0.080512,
        0.154788,
        0.225563,
        0.29071,
        0.34728,
        0.39219,
        0.42847,
        0.46181,
        0.49445,
        0.52704,
        0.55928,
        0.59087,
        0.62214,
        0.65244,
        0.68234,
        0.75483,
        0.82289,
        0.88814,
        0.95123,
        1.07176,
        1.18436,
        1.39152,
        1.58006,
        1.91444,
        2.21131,
    ]
)
U_R_P_err = np.concatenate([np.full(14, 0.0001), np.full(9, 0.001), np.full(2, 0.002)])

U_0_D = np.array([10.1, 9.01, 8.0, 7.01, 6.01, 5.01, 4.01, 3.01, 2.01])
U_0_D_err = U_0_D * 0.0005 + 0.01

U_R_D = np.array([6.760, 6.130, 5.600, 4.600, 4.330, 3.170, 3.000, 1.844, 1.418])
U_R_D_err = np.array([0.12, 0.12, 0.1, 0.08, 0.08, 0.06, 0.05, 0.02, 0.01])

lines = np.array(
    [
        [[6.760, 6.740, 6.680, 6.640, 6.600], [3.040, 3.240, 3.560, 3.920, 4.080]],
        [[6.130, 6.070, 6.010, 5.970, 5.930], [7.160, 7.800, 8.280, 8.760, 9.120]],
        [[5.600, 5.540, 5.500, 5.420, 5.360], [5.600, 6.400, 7.000, 8.000, 9.000]],
        [[4.600, 4.560, 4.520, 4.480, 4.440], [7.400, 8.200, 9.000, 9.800, 10.60]],
        [[4.330, 4.270, 4.230, 4.190, 4.130], [8.400, 10.00, 11.60, 12.80, 14.80]],
        [[3.170, 3.130, 3.110, 3.090, 3.030], [10.00, 11.60, 13.20, 14.80, 16.40]],
        [[3.000, 2.980, 2.960, 2.880, 2.840], [12.00, 14.40, 16.40, 24.80, 30.00]],
        [[1.844, 1.830, 1.816, 1.804, 1.792], [20.40, 23.60, 26.00, 29.20, 32.80]],
        [[1.418, 1.410, 1.390, 1.386, 1.383], [26.40, 31.20, 38.80, 42.80, 46.00]],
    ]
)

lines_err = np.array([0.12, 0.12, 0.1, 0.08, 0.08, 0.06, 0.05, 0.02, 0.01])

for i, row in enumerate(lines):
    lines[i][1] = row[1] * 0.001


def mean_and_error(input):
    l = len(input)
    me = np.mean(input)
    error = np.sqrt(np.sum((input - me) ** 2) / (l * (l - 1)))
    return (me, error)


def power_loss(U_0, U_R, R):
    return (U_0 - U_R) * U_R / R


def power_loss_err(U_0, U_R, R, U_0_err, U_R_err, R_err):
    U_0_del = U_R / R * U_0_err
    U_R_del = (U_0 - 2 * U_R) / R * U_R_err
    R_del = (U_0 - U_R) / (R**2) * U_R * R_err
    return np.sqrt(U_0_del**2 + U_R_del**2 + R_del**2)


def R_w(U_0, U_R, R):
    return (U_0 - U_R) / U_R * R


def R_w_err(U_0, U_R, R, U_0_err, U_R_err, R_err):
    U_0_del = R / U_R * U_0_err
    U_R_del = U_0 * R / (U_R**2) * U_R_err
    R_del = (U_0 / U_R - 1) * R_err
    return np.sqrt(U_0_del**2 + U_R_del**2 + R_del**2)


def line(params, x):
    return params[0] * x + params[1]


def new_line(x, a, b):
    return a * x + b


def fit_model(x, y, x_err, y_err):
    model = scipy.odr.Model(line)
    data = scipy.odr.RealData(x, y, sx=x_err, sy=y_err)
    odr_instance = scipy.odr.ODR(data, model, beta0=[1, 1])
    output = odr_instance.run()
    return (output.beta, output.sd_beta)


def find_lines(measurements, errors):
    parameters = []
    for i, series in enumerate(measurements):
        parameters.append(
            scipy.optimize.curve_fit(new_line, series[1], series[0], sigma=errors[i])
        )
    return parameters


def X(U_0, U_R, R, R_0, der):
    return -(R * U_0 * der) / (ALPHA * R_0 * (U_R**2))


def X_err(X, U_0, U_R, R, R_0, der, U_0_err, U_R_err, R_err, R_0_err, der_err):
    del_R = (X * R_err / R) ** 2
    del_der = (X * der_err / der) ** 2
    del_U_0 = (X * U_0_err / U_0) ** 2
    del_R_0 = (X * R_0_err / R_0) ** 2
    del_U_R = (2 * X * U_R_err / U_R) ** 2
    return np.sqrt(del_R + del_der + del_U_0 + del_R_0 + del_U_R)


def Y(U_0, U_R, R, a, b):
    powe = (U_0 - U_R) * U_R / R
    fit = a * (U_0 - U_R) / U_R * R + b
    return powe - fit


def Y_err(U_0, U_R, R, a, b, U_0_err, U_R_err, R_err, a_err, b_err):
    del_U_0 = ((U_R / R - a * R / U_R) * U_0_err) ** 2
    del_U_R = (((U_0 - 2 * U_R) / R + a * U_0 / (U_R**2)) * U_R_err) ** 2
    del_R = (((U_0 - U_R) * U_R / (R**2) + a * (U_0 - U_R) / U_R) * R_err) ** 2
    del_a = (((U_0 - U_R) / U_R * R) * a_err) ** 2
    return np.sqrt(del_U_0 + del_U_R + del_R + del_a + (b_err**2))


def Resistor_measurment_error(R):
    resist_range = 200
    return (0.03 * R + resist_range * 0.005) * 0.01


def combined_error(stat, measure):
    return np.sqrt(stat**2 + measure**2)


R_0_mean, R_0_err_stat = mean_and_error(R_0)
R_0_err_measure = Resistor_measurment_error(R_0_mean)
R_mean, R_err_stat = mean_and_error(R)
R_err_measure = Resistor_measurment_error(R_mean)

R_0_err = combined_error(R_0_err_stat, R_0_err_measure)
R_err = combined_error(R_err_stat, R_err_measure)

power_loss_record = power_loss(U_0_P, U_R_P, R_mean)
power_loss_record_error = power_loss_err(
    U_0_P, U_R_P, R_mean, U_0_P_err, U_R_P_err, R_err
)
bulb_resistance = R_w(U_0_P, U_R_P, R_mean)
bulb_resistance_error = R_w_err(U_0_P, U_R_P, R_mean, U_0_P_err, U_R_P_err, R_err)

fited_lines = find_lines(lines, U_R_D_err)

derivatives = []
derivatives_err = []
for i in fited_lines:
    derivatives.append(i[0][0])
    derivatives_err.append(np.sqrt(i[1][0][0]))

print(
    f"resistor resistance: {R_mean:.6f}, stat error: {R_err_stat:.6f}, measure error: {R_err_measure:.6f} full error: {R_err:.6f}"
)
print(
    f"bulb default resistance: {R_0_mean:.6f}, stat error: {R_0_err_stat:.6f}, measure error: {R_0_err_measure:.6f} full error: {R_0_err:.6f}"
)
s = 1
e = 8
model_fit, model_fit_err = fit_model(
    bulb_resistance[s:e],
    power_loss_record[s:e],
    bulb_resistance_error[s:e],
    power_loss_record_error[s:e],
)
R_fited = -model_fit[1] / model_fit[0]
R_fited_err = np.sqrt(
    (model_fit_err[1] / model_fit[0]) ** 2
    + (model_fit[1] / (model_fit[0] ** 2) * model_fit_err[0]) ** 2
)
print(f"fited parameters: {model_fit}, parameters error: {model_fit_err}")
# R_0_mean = R_fited
# R_0_err = R_fited_err

final_X = X(U_0_D, U_R_D, R_mean, R_0_mean, derivatives)
final_X_err = X_err(
    final_X,
    U_0_D,
    U_R_D,
    R_mean,
    R_0_mean,
    derivatives,
    U_0_D_err,
    U_R_D_err,
    R_err,
    R_0_err,
    derivatives_err,
)
final_Y = Y(U_0_D, U_R_D, R_mean, *model_fit)
final_Y_err = Y_err(
    U_0_D, U_R_D, R_mean, *model_fit, U_0_D_err, U_R_D_err, R_err, *model_fit_err
)

final_model, final_model_err = fit_model(final_X, final_Y, final_X_err, final_Y_err)
print(f"fited resistance: {R_fited}, fited resistance error: {R_fited_err}")
print(
    f"fited final parameters: {final_model}, final parameters error: {final_model_err}"
)
print(
    f"final mass*heatcapa: {final_model[0]}, final mass*heatcapa error: {final_model_err[0]}"
)
print(f"final mass: {final_model[0]/CAPA}, final mass error: {final_model_err[0]/CAPA}")


def plot_power(start=0, end=len(bulb_resistance), params=None):
    if params is not None:
        x_model = np.linspace(
            min(bulb_resistance[start:end]), max(bulb_resistance[start:end]), 100
        )
        y_model = line(params, x_model)
        plt.plot(x_model, y_model, "r-", label="Dopasowanie Liniowe")
    plt.errorbar(
        bulb_resistance[start:end],
        power_loss_record[start:end],
        xerr=bulb_resistance_error[start:end],
        yerr=power_loss_record_error[start:end],
        fmt="o",
        label="Punkty pomiarowe",
    )
    plt.xlabel("Opór włókna $R_w$ [Ω]")
    plt.ylabel("Rozpraszana energia $P_s$ [W]")
    plt.legend()
    plt.savefig("power_plot.png", dpi=300, bbox_inches="tight")


def test_fit(ind):
    x_data = 1000*lines[ind][1]
    y_data = lines[ind][0]
    x_model = np.linspace(min(x_data), max(x_data))
    params = fited_lines[ind][0]
    y_model = new_line(x_model, params[0]/1000, params[1])
    plt.errorbar(x_data, y_data, yerr=lines_err[ind], fmt="o", label="Punkty pomiarowe")
    plt.plot(x_model, y_model, "r-", label="Dopasowanie Liniowe")
    plt.ylabel(r"$U_R$ [mV]")
    plt.xlabel(r"$t$ [ms]")
    plt.legend()
    plt.savefig(f"nachylenie_{ind+1}.png", dpi=300, bbox_inches="tight")
    plt.clf()


def final_graph():
    x_model = np.linspace(min(final_X), max(final_X))
    y_model = line(final_model, x_model)
    plt.errorbar(
        final_X,
        final_Y,
        xerr=final_X_err,
        yerr=final_Y_err,
        fmt="o",
        label="Punkty pomiarowe",
    )
    plt.plot(x_model, y_model, label="Dopasowanie liniowe")
    plt.xlabel("Zmienna pomocnicza X [K/s]")
    plt.ylabel("Zmienna pomocnicza Y [W]")
    plt.legend()
    plt.savefig("final_graph.png", dpi=300, bbox_inches="tight")
    plt.clf()


#plot_power()
# plot_power(s, e, model_fit)
# final_graph()
for i in range(len(lines)):
    test_fit(i)
