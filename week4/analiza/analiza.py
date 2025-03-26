import scipy
import matplotlib.pyplot as plt
import numpy as np

MILI = 1e-3
CENTY = 1e-2
KELVIN = 273.15
TEMP_ERROR = 1
DISTANCE_ERROR = 1*CENTY

cube_measurements = np.array([
    [ 50,  55,  60,  65,  70,  75,  80,  85,  90,  95, 100, 105, 110, 115, 120],  # T [C]
    [2.05, 2.56, 2.94, 3.45, 3.89, 4.43, 4.97, 5.43, 5.95, 6.52, 7.12, 7.66, 8.34, 8.87, 9.52],  # U_czarna [mV]
    [2.05, 2.56, 2.93, 3.41, 3.89, 4.40, 4.94, 5.41, 5.85, 6.48, 7.06, 7.63, 8.32, 8.85, 9.52],  # U_biała [mV]
    [0.17, 0.18, 0.23, 0.25, 0.26, 0.29, 0.32, 0.34, 0.38, 0.42, 0.45, 0.50, 0.54, 0.58, 0.63],  # U_metal_błyszczący [mV] 
    [0.53, 0.63, 0.71, 0.80, 0.95, 1.04, 1.16, 1.28, 1.39, 1.53, 1.71, 1.85, 2.06, 2.15, 2.34],  # U_metal_matowy [mV] 
    [0.15, 0.15, 0.15, 0.17, 0.21, 0.19, 0.17, 0.14, 0.14, 0.17, 0.19, 0.21, 0.24, 0.26, 0.25]   # U_otoczenie [mV] 
])
cube_measurements[1:] = MILI*cube_measurements[1:]
cube_measurements[0] = KELVIN+cube_measurements[0]

REFERENCE_TEMP = 300
REFERENCE_RESISTANCE = 0.277

BULB_DISTANCE = 1.54 # d [m]
SENSOR_DISTANCE = 1.49 # d [m]
temp_measurments = np.array([
    [0.828, 1.775, 2.725, 3.681, 4.64, 5.60, 6.57, 7.53, 8.50, 9.47],  # U_żarówka [V] 
    [0.919, 1.191, 1.432, 1.648, 1.847, 2.033, 2.207, 2.370, 2.524, 2.671],  # I_żarówka [A]
    [0.15,  1.13, 2.89, 5.68, 9.12, 12.31, 16.15, 20.25, 25.13, 30.24],  # U_czujnik [mV] 
    [0.05,  0.05, 0.05, 0.07, 0.08, 0.06, 0.09, 0.08, 0.10, 0.11]  # U_otoczenie [mV]
])
temp_measurments[2:] = MILI*temp_measurments[2:]

DISTANCE_BULB_VOLTAGE = 9.47
DISTANCE_BULB_CURRENT = 2.670
distance_measurments = np.array([
    [149, 144, 139, 134, 129, 124, 119, 114, 109],        # d [cm]
    [30.58, 9.29, 4.11, 2.36, 1.52, 1.08, 0.82, 0.64, 0.52],  # U_czujnik [mV]
    [0.12, 0.10, 0.11, 0.11, 0.11, 0.08, 0.06, 0.05, 0.05]   # U_otoczenie [mV]
])
distance_measurments[1:] = MILI*distance_measurments[1:]
distance_measurments[0] = CENTY*distance_measurments[0]

def MeanAndStatisticalError(x):
    mean = np.mean(x);
    stat_err = np.sqrt(np.sum((x-mean)**2)/len(x))
    return mean, stat_err

def CombinedError(stat, measur):
    return np.sqrt(stat**2 + measur**2/3)

def DetectorVoltageError(voltage):
    return voltage*0.0012+MILI*0.02

def BulbVoltageError(voltage):
    voltage_err = 0.005*voltage
    voltage_err[voltage > 4] += 0.03
    voltage_err[voltage <= 4] += 0.003

    return np.abs(voltage_err)

def BulbCurrentError(current):
    return 0.02*current+0.005

def BackgroundVoltage(measurments):
    mean_surrounding_voltage, mean_surrounding_voltage_staterr = MeanAndStatisticalError(measurments)
    mean_surrounding_voltage_measerr = DetectorVoltageError(mean_surrounding_voltage)
    mean_surrounding_voltage_err = CombinedError(mean_surrounding_voltage_staterr, DetectorVoltageError(mean_surrounding_voltage))
    print(f"mean surronding voltage: {mean_surrounding_voltage}, statistical error: {mean_surrounding_voltage_staterr}, measurment error: {mean_surrounding_voltage_measerr}, error: {mean_surrounding_voltage_err}")
    return mean_surrounding_voltage, mean_surrounding_voltage_err

def LinearModel(params, x):
    m, c = params
    return m*x+c

def FitToLinearModel(x, y, x_err, y_err):
    linear = scipy.odr.Model(LinearModel)
    data = scipy.odr.RealData(x, y, sx=x_err, sy=y_err)
    initial_guess = [1.0, 0.0]

    odr = scipy.odr.ODR(data, linear, beta0=initial_guess)
    out = odr.run()
    return out.beta, out.sd_beta

def PlotLineFit(x, y, x_err, y_err, params, x_label, y_label, data_label, fit_label):
    plt.errorbar(x, y, xerr=x_err, yerr=y_err, fmt='o', label=data_label)
    x_fit = np.linspace(min(x), max(x), 200)
    y_fit = LinearModel(params, x_fit)
    plt.plot(x_fit, y_fit, 'r-', label=fit_label)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.show()

def AdjustVoltage(voltage, adjustment, adjustment_err):
    voltage_adj = voltage-adjustment;
    voltage_adj_err = np.sqrt(DetectorVoltageError(voltage)**2+adjustment_err**2)
    return voltage_adj, np.abs(voltage_adj_err)

def CubeLineFit(temp, voltage, adjustment, adjustment_err):
    voltage_adj, voltage_adj_err = AdjustVoltage(voltage, adjustment, adjustment_err)
    temp_adj = temp**4
    temp_adj_err = 4*temp**3*TEMP_ERROR
    params_out, params_err = FitToLinearModel(temp_adj, voltage_adj, temp_adj_err, voltage_adj_err)
    print(f"fited line params: {params_out}, error: {params_err}")
    PlotLineFit(temp_adj, voltage_adj, temp_adj_err, voltage_adj_err, params_out, "", "", "", "")

def CubeAnalysis():
    mean_surrounding_voltage, mean_surrounding_voltage_err = BackgroundVoltage(cube_measurements[-1])
    CubeLineFit(cube_measurements[0], cube_measurements[1], mean_surrounding_voltage, mean_surrounding_voltage_err)
    CubeLineFit(cube_measurements[0], cube_measurements[2], mean_surrounding_voltage, mean_surrounding_voltage_err)
    CubeLineFit(cube_measurements[0], cube_measurements[3], mean_surrounding_voltage, mean_surrounding_voltage_err)
    CubeLineFit(cube_measurements[0], cube_measurements[4], mean_surrounding_voltage, mean_surrounding_voltage_err)

def Resistance(voltage, current, voltage_err, current_err):
    resistance = voltage/current
    resistance_err = resistance*np.sqrt((voltage_err/voltage)**2+(current_err/current)**2)
    return resistance, np.abs(resistance_err)

def Alpha(resistance, resistance_err):
    exp = 0.11778
    alpha = 0.00407*((resistance/REFERENCE_RESISTANCE)**exp)
    alpha_err = exp*alpha*resistance_err/resistance
    return alpha, np.abs(alpha_err)

def BulbTemperature(resistance, resistance_err):
    alpha = Alpha(resistance, resistance_err)
    temp = (resistance - REFERENCE_RESISTANCE)/(alpha[0]*REFERENCE_RESISTANCE)+REFERENCE_TEMP
    temp_err = np.sqrt((resistance_err/(alpha[0]*REFERENCE_RESISTANCE))**2+(resistance*alpha[1]/(alpha[0]**2*REFERENCE_RESISTANCE))**2)
    return temp, np.abs(temp_err)

def TemperatureAnalysis():
    temp, temp_err = BulbTemperature(*Resistance(temp_measurments[0], temp_measurments[1], BulbVoltageError(temp_measurments[0]), BulbCurrentError(temp_measurments[1])))
    mean_background_voltage, mean_background_voltage_err = BackgroundVoltage(temp_measurments[-1])
    voltage_adj, voltage_adj_err = AdjustVoltage(temp_measurments[2], mean_background_voltage, mean_background_voltage_err)
    temp_log = np.log(temp)
    temp_log_err = np.abs(temp_err/temp)
    voltage_adj_log = np.log(voltage_adj)
    voltage_adj_log_err = np.abs(voltage_adj_err/voltage_adj)
    params_out, params_err = FitToLinearModel(temp_log, voltage_adj_log, temp_log_err, voltage_adj_log_err)
    print(f"fited line params: {params_out}, error: {params_err}")
    PlotLineFit(temp_log, voltage_adj_log, temp_log_err, voltage_adj_log_err, params_out, "", "", "", "")

def InverseSquareLaw(distance, distance_err):
    distance_adj = 1/distance**2
    distance_adj_err = 2*distance_err/(distance_adj**3)
    return distance_adj, distance_adj_err

def DistanceAnalysis():
    true_distance = np.abs(distance_measurments[0]-BULB_DISTANCE)
    distance_adj = np.log(true_distance)
    distance_adj_err = np.abs(DISTANCE_ERROR/true_distance)
    mean_background_voltage, mean_background_voltage_err = BackgroundVoltage(distance_measurments[-1])
    voltage_adj, voltage_adj_err = AdjustVoltage(distance_measurments[1], mean_background_voltage, mean_background_voltage_err)
    voltage_adj_log = np.log(voltage_adj)
    voltage_adj_log_err = np.abs(voltage_adj_err/voltage_adj)
    params_out, params_err = FitToLinearModel(distance_adj, voltage_adj_log, distance_adj_err, voltage_adj_log_err)
    print(f"fited line params: {params_out}, error: {params_err}")
    PlotLineFit(distance_adj, voltage_adj_log, distance_adj_err, voltage_adj_log_err, params_out, "", "", "", "")

CubeAnalysis()
# TemperatureAnalysis()
# DistanceAnalysis()

