import numpy as np
import scipy
import matplotlib.pyplot as plt

MILI = 0.001
SAMPLE_HEIGHT = 1*MILI # [m]
SAMPLE_WIDTH = 10*MILI # [m]
SAMPLE_LENGTH = 20*MILI # [m]
HALL_RANGE = 0.2
KELVINS = 273.15
BOLTZMAN_CONST = 1.380649e-23
ELECTRON_CHARGE = 1.602176634e-19

ohm_measurments = np.array([[-26.82, -20.08, -24.74, -19.73, -9.69, -4.70],  #I [mA]
                            [-0.955, -0.714, -0.882, -0.704, -0.345, -0.167], #U [V]
                            ])
ohm_measurments[0] *= MILI

const_current_value = -26.82*MILI # [A]
const_current_measurments = np.array([[53.3, 46.3, 38.8, 32.5, 18.5, 6.7, 63.8, 57.5], #B+ [mT]
                          [56.8, 49.3, 42.5, 35.3, 22.0, 7.5, 67.4, 60.5], #B- [mT]
                          [-7.442, -6.457, -5.430, -4.625, -2.572, -0.800, -8.795, -7.988], #U M+ P+ [mV]
                          [8.130, 7.123, 6.159, 5.158, 3.245, 1.183, 9.542, 8.665], #U M- P+ [mV]
                          [7.428, 6.454, 5.424, 4.520, 2.560, 0.785, 8.789, 7.982], #U M+ P- [mV]
                          [-8.148, -7.145, -6.148, -5.161, -3.256, -1.196, -9.565, -8.688]]) #U M- P- [mV]
const_current_measurments *= MILI

const_magnetic_field_value_p = 67.4*MILI # [T]
const_magnetic_field_value_m = 63.8*MILI # [T]
const_magnetic_field_measurments = np.array([[-26.82, -21.71, -16.77, -11.82, -6.82, -28.94, -2.70], #I+ [mA]
                                             [26.82, 21.71, 16.77, 11.82, 6.82, 28.94, 2.70], #I- [mA]
                                             [-8.818, -7.129, -5.517, -3.882, -2.247, -9.532, -0.898], #U [mV] M+ P+
                                             [9.548, 7.722, 5.952, 4.188, 2.419, 10.275, 0.968], #U [mV] M- P+
                                             [8.800, 7.142, 5.512, 3.881, 2.242, 9.511, 0.899], #U [mV] M+ P-
                                             [-9.561, -7.741, -5.963, -4.192, -2.426, -10.287, -0.972]]) #U [mV] M- P-
const_magnetic_field_measurments *= MILI

temp_magnetic_field = 63.8*MILI # [T]
temp_current_sample = -26.88*MILI # [A]
cooling_measurments = np.array([[143.8, 136.2, 127.3, 120.3, 112.1, 105.3, 97.5, 86.6, 75.3, 66.2, 55.5, 44.2, 36.2], #T [deg]
                                [-1.117, -1.402, -1.773, -2.183, -2.800, -3.408, -4.201, -5.452, -6.505, -7.219, -7.825, -8.267, -8.504], #U_H [mV]
                                [-0.410, -0.506, -0.627, -0.742, -0.891, -1.018, -1.147, -1.258, -1.281, -1.255, -1.159, -1.112, -1.056]]) #U_P [V]
cooling_measurments = cooling_measurments[:, ::-1]
cooling_measurments[1] *= MILI
                                
heating_measurments = np.array([[34.6, 45.2, 54.3, 65.1, 75.1, 85.2, 96.7, 104.8, 118.5, 126.7, 134.3, 140.6, 145.4], #T [deg]
                                [-8.521, -8.130, -7.776, -7.190, -6.443, -5.488, -4.235, -3.424, -2.362, -1.873, -1.498, -1.187, -1.037], #U_H [mV]
                                [-1.047, -1.126, -1.188, -1.253, -1.287, -1.265, -1.150, -1.017, -0.757, -0.623, -0.510, -0.435, -0.382]]) #U_P [V]
heating_measurments[1] *= MILI

def MultimeterVoltageError(val, range_val):
    return val*0.00015+range_val*0.00004

def HallVoltageConstCurrent(measurments):
    hall_voltage = (measurments[2]-measurments[3]-measurments[4]+measurments[5])/4
    hall_voltage_err = np.max(np.array([MultimeterVoltageError(measurments[2], HALL_RANGE),
                                     MultimeterVoltageError(measurments[3], HALL_RANGE),
                                     MultimeterVoltageError(measurments[4], HALL_RANGE),
                                     MultimeterVoltageError(measurments[5], HALL_RANGE)]), axis=0)
    return hall_voltage, hall_voltage_err

def Magnetic(first, second):
    return (first+second)/2

def MagneticError(val):
    return np.abs(val*0.05)

def MagneticFielsTotal(measurments):
    field = Magnetic(measurments[0], measurments[1])
    err = np.max(np.array([MagneticError(measurments[0]), MagneticError(measurments[1])]), axis=0)
    return field, err

def TempError(temp):
    return temp*0.001+1

def CurrentError(val):
    return np.abs(val*0.02)

def SampleVoltageError(val):
    return np.abs(val*0.005)

def Current(measurments):
    err = CurrentError(measurments[0])
    return measurments[0], err

def HallVoltageXInput(current, field):
    return current*field

def HallVoltageXInputError(current, current_err, field, field_err):
    return np.sqrt((current*field_err)**2+(current_err*field)**2)

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

def TempMeasurmentsExp(temp, exp):
    temp_reduced = temp**exp
    temp_reduced_err = np.abs(exp*TempError(temp)*(temp**(exp-1)))
    return temp_reduced, temp_reduced_err

def Conductivity(voltag, current, voltage_err, current_err):
    cond = (SAMPLE_LENGTH*current)/(SAMPLE_WIDTH*SAMPLE_HEIGHT*voltag)
    cond_err = SAMPLE_LENGTH/(SAMPLE_HEIGHT*SAMPLE_WIDTH)*np.sqrt((current_err/voltag)**2+(current*voltage_err/(voltag**2))**2)
    return cond, cond_err

def TempMobility(voltage, hall_voltage, field, voltage_err, hall_voltage_err, field_err):
    mobility = hall_voltage/(field*voltage)*SAMPLE_LENGTH/SAMPLE_WIDTH
    mobility_err = mobility*np.sqrt((hall_voltage_err/hall_voltage)**2+(field_err/field)**2+(voltage_err/voltage)**2)
    return mobility, mobility_err

def TempConcentration(hall_voltage, field, current, hall_voltage_err, field_err, current_err):
    concentration = current*field/(SAMPLE_HEIGHT*ELECTRON_CHARGE*hall_voltage)
    concentration_err = concentration*np.sqrt((current_err/current)**2+(field_err/field)**2+(hall_voltage_err/hall_voltage)**2)
    return concentration, concentration_err

def PlotLineFit(x, y, x_err, y_err, params, x_label, y_label, data_label, fit_label, title):
    plt.errorbar(x, y, xerr=x_err, yerr=y_err, fmt='o', label=data_label)
    x_fit = np.linspace(min(x), max(x), 200)
    y_fit = LinearModel(params, x_fit)
    plt.plot(x_fit, y_fit, 'r-', label=fit_label)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    #plt.title(title)
    plt.legend()
    plt.show()

def PlotTemperatures(cooling, heating):
    plt.errorbar(cooling[0], cooling[1], xerr=TempError(cooling[0]), yerr=MultimeterVoltageError(cooling[1], HALL_RANGE),fmt='bv', label="Chłodzenie próbki")
    plt.errorbar(heating[0], heating[1], xerr=TempError(heating[0]), yerr=MultimeterVoltageError(heating[1], HALL_RANGE), fmt='r^', label="Grzanie próbki")
    plt.xlabel(r"$T \ [^\circ C]$")
    plt.ylabel(r"$U_H \ [mV]$")
    plt.legend()
    plt.show()

def PlotTemperaturesConductivity(cooling, heating):
    plt.errorbar(cooling[0], 1/cooling[1], xerr=TempError(cooling[0]), yerr=MultimeterVoltageError(cooling[1], HALL_RANGE)/(cooling[1]**2),fmt='bv', label="Chłodzenie próbki")
    plt.errorbar(heating[0], 1/heating[1], xerr=TempError(heating[0]), yerr=MultimeterVoltageError(heating[1], HALL_RANGE)/(heating[1]**2), fmt='r^', label="Grzanie próbki")
    plt.xlabel(r"$T \ [^\circ C]$")
    plt.ylabel(r"$U_H \ [mV]$")
    plt.legend()
    plt.show()

def PlotTemperaturesConductivityAdjusted(cooling, heating):
    conductivity_cooling, conductivity_cooling_err = Conductivity(cooling[2], temp_current_sample, SampleVoltageError(cooling[2]), CurrentError(temp_current_sample))
    conductivity_heating, conductivity_heating_err = Conductivity(heating[2], temp_current_sample, SampleVoltageError(heating[2]), CurrentError(temp_current_sample))
    plt.errorbar(1/cooling[0], np.log(conductivity_cooling), xerr=TempError(cooling[0])/(cooling[0]**2), yerr=conductivity_cooling_err/conductivity_cooling,fmt='bv', label="Chłodzenie próbki")
    plt.errorbar(1/heating[0], np.log(conductivity_heating), xerr=TempError(heating[0])/(heating[0]**2), yerr=conductivity_heating_err/conductivity_heating, fmt='r^', label="Grzanie próbki")
    plt.xlabel(r"$T^{-1} \ [(^\circ C)^{-1}]$")
    plt.ylabel(r"$\log(\sigma)$")
    plt.legend()
    plt.show()

def PlotTemperaturesMobility(cooling, heating, field, field_err):
    mobility_cooling, mobility_cooling_err = TempMobility(cooling[2], cooling[1], field, SampleVoltageError(cooling[2]), MultimeterVoltageError(cooling[1], HALL_RANGE), field_err)
    mobility_heating, mobility_heating_err = TempMobility(heating[2], heating[1], field, SampleVoltageError(heating[2]), MultimeterVoltageError(heating[1], HALL_RANGE), field_err)
    plt.errorbar(cooling[0], mobility_cooling, xerr=TempError(cooling[0]), yerr=mobility_cooling_err, fmt='bs', label="Chłodzenie próbki")
    plt.errorbar(heating[0], mobility_heating, xerr=TempError(heating[0]), yerr=mobility_heating_err, fmt='rs', label="Grzanie próbki")
    plt.xlabel(r"$T \ [^{\circ}C]$")
    plt.ylabel(r"$\mu \ [cm^2s^{-1}V^{-1}]$")
    plt.legend()
    plt.show()

def PlotTemperaturesConcentration(cooling, heating, field, current, field_err, current_err):
    concentartion_cooling, concentartion_cooling_err = TempConcentration(cooling[1], field, current, MultimeterVoltageError(cooling[1], HALL_RANGE), field_err, current_err)
    concentartion_heating, concentartion_heating_err = TempConcentration(heating[1], field, current, MultimeterVoltageError(heating[1], HALL_RANGE), field_err, current_err)
    plt.errorbar(cooling[0], concentartion_cooling, xerr=TempError(cooling[0]), yerr=concentartion_cooling_err, fmt='bs', label="Chłodzenie próbki")
    plt.errorbar(heating[0], concentartion_heating, xerr=TempError(heating[0]), yerr=concentartion_heating_err, fmt='rs', label="Grzanie próbki")
    plt.xlabel(r"$T \ [^{\circ}C]$")
    plt.ylabel(r"$n \ [cm^{-3}]$")
    plt.legend()
    plt.show()

def PlotTemperaturesVoltage(cooling, heating):
    plt.errorbar(cooling[0], cooling[2], xerr=TempError(cooling[0]), yerr=SampleVoltageError(cooling[2]),fmt='bv', label="Chłodzenie próbki")
    plt.errorbar(heating[0], heating[2], xerr=TempError(heating[0]), yerr=SampleVoltageError(heating[2]), fmt='r^', label="Grzanie próbki")
    plt.xlabel(r"$T \ [^\circ C]$")
    plt.ylabel(r"$U \ [V]$")
    plt.legend()
    plt.show()


def PlotTemperaturesLog(cooling, heating):
    plt.errorbar(np.exp(-1/cooling[0]), cooling[1], xerr=np.abs(np.exp(-1/cooling[0])*TempError(cooling[0])/(cooling[0]**2)), yerr=MultimeterVoltageError(cooling[1], HALL_RANGE),fmt='bv', label="Chłodzenie próbki")
    plt.errorbar(np.exp(-1/heating[0]), heating[1], xerr=np.abs(np.exp(-1/heating[0])*TempError(heating[0])/(heating[0]**2)), yerr=MultimeterVoltageError(heating[1], HALL_RANGE), fmt='r^', label="Grzanie próbki")
    plt.xlabel(r"$\exp(T^{-1})$")
    plt.ylabel(r"$U_H \ [mV]$")
    plt.legend()
    plt.show()

def PlotTemperaturesExp(cooling, heating, exp):
    adjusted_cooling_temp, adjusted_cooling_temp_err = TempMeasurmentsExp(cooling[0], exp)
    adjusted_heating_temp, adjusted_heating_temp_err = TempMeasurmentsExp(heating[0], exp)
    plt.errorbar(adjusted_cooling_temp, cooling[1], xerr=adjusted_cooling_temp_err, yerr=MultimeterVoltageError(cooling[1], HALL_RANGE),fmt='bv', label="Chłodzenie próbki")
    plt.errorbar(adjusted_heating_temp, heating[1], xerr=adjusted_heating_temp_err, yerr=MultimeterVoltageError(heating[1], HALL_RANGE), fmt='r^', label="Grzanie próbki")
    plt.xlabel(r"$T^{-\frac{3}{2}} \ [(^\circ C)^{-\frac{3}{2}}]$")
    plt.ylabel(r"$U_H \ [mV]$")
    plt.legend()
    plt.show()

const_current_hall_voltage, const_current_hall_voltage_err = HallVoltageConstCurrent(const_current_measurments)
const_current_field, const_current_field_err = MagneticFielsTotal(const_current_measurments)
# print(const_current_hall_voltage_err)
# print(const_current_field_err)
const_current_input = HallVoltageXInput(const_current_value, const_current_field)
const_current_input_err = HallVoltageXInputError(const_current_value, CurrentError(const_current_value), const_current_field, const_current_field_err)


const_field_hall_voltage, const_field_hall_voltage_err = HallVoltageConstCurrent(const_magnetic_field_measurments)
const_field_current, const_field_current_err = Current(const_magnetic_field_measurments)
# print(const_field_hall_voltage_err)
# print(const_field_current_err)
const_field_input = HallVoltageXInput(const_field_current, Magnetic(const_magnetic_field_value_p, const_magnetic_field_value_m))
const_field_input_err = HallVoltageXInputError(const_field_current, const_field_current_err, Magnetic(const_magnetic_field_value_p, const_magnetic_field_value_m), np.max(np.array([MagneticError(const_magnetic_field_value_p), MagneticError(const_magnetic_field_value_m)]), axis=0))

const_current_params, const_current_params_err = FitToLinearModel(const_current_input, const_current_hall_voltage, const_current_input_err, const_current_hall_voltage_err)
const_field_params, const_field_params_err = FitToLinearModel(const_field_input, const_field_hall_voltage, const_field_input_err, const_field_hall_voltage_err)

const_current_hall_constant = const_current_params*SAMPLE_HEIGHT
const_current_hall_constant_err = const_current_params_err*SAMPLE_HEIGHT
const_field_hall_constant = const_field_params*SAMPLE_HEIGHT
const_field_hall_constant_err = const_field_params_err*SAMPLE_HEIGHT

resistance_params, resistance_params_err = FitToLinearModel(ohm_measurments[0], ohm_measurments[1], SampleVoltageError(ohm_measurments[0]), CurrentError(ohm_measurments[1]))
conductivity = SAMPLE_LENGTH/(SAMPLE_HEIGHT*SAMPLE_WIDTH*resistance_params[0])
conductivity_err = resistance_params_err[0]*SAMPLE_LENGTH/(SAMPLE_HEIGHT*SAMPLE_WIDTH*(resistance_params[0]**2)) 
# print(f"fited resistance value: {resistance_params}, error: {resistance_params_err}")
# print(f"fited conductivit value: {conductivity}, error: {conductivity_err}")
# PlotLineFit(ohm_measurments[0]/MILI,
#             ohm_measurments[1]/MILI, 
#             SampleVoltageError(ohm_measurments[0])/MILI, 
#             CurrentError(ohm_measurments[1])/MILI, 
#             (resistance_params[0], resistance_params[1]/MILI), 
#             r"$I \ [mA]$", 
#             r"$U \ [mV]$", 
#             "punkty pomiarowe", 
#             "dopasowanie liniowe", 
#             "Zależność napięcia od natężenia na próbce")
# 
# print(f"current used: {const_current_value}, error:{CurrentError(const_current_value)}")
# print(f"parameters to constant current value: {const_current_params}, error: {const_current_params_err}")
# print(f"hall constant to constant current value: {const_current_hall_constant}, error: {const_current_hall_constant_err}")
# PlotLineFit(const_current_input/MILI,
#             const_current_hall_voltage/MILI, 
#             const_current_input_err/MILI, 
#             const_current_hall_voltage_err/MILI, 
#             (const_current_params[0], const_current_params[1]/MILI), 
#             r"$BI \, [mTA]$", 
#             r"$U_{H} \, [mV]$", 
#             "punkty pomiarowe", 
#             "dopasowanie liniowe", 
#             "Zależność napięcia halla od wartości iloczynu pola magnetycznego i natężenia \ndla stałego natężenia.")
# 
# print(f"magnetic field used: {(const_magnetic_field_value_p+const_magnetic_field_value_m)/2}, error:{max([MagneticError(const_magnetic_field_value_p), MagneticError(const_magnetic_field_value_m)])}")
# print(f"parameters to constant field value: {const_field_params}, error: {const_field_params_err}")
#print(f"hall constant to constant field value: {const_field_hall_constant}, error: {const_field_hall_constant_err}")
# PlotLineFit(const_field_input/MILI,
#             const_field_hall_voltage/MILI, 
#             const_field_input_err/MILI, 
#             const_field_hall_voltage_err/MILI, 
#             (const_field_params[0], const_field_params[1]/MILI), 
#             r"$BI \, [mTA]$", 
#             r"$U_{H} \, [mV]$", 
#             "punkty pomiarowe", 
#             "dopasowanie liniowe", 
#             "Zależność napięcia halla od wartości iloczynu pola magnetycznego i natężenia \ndla stałego pola magnetycznego.")
# 
final_hall_const = (const_current_hall_constant/(const_current_hall_constant_err**2)+const_field_hall_constant/(const_field_hall_constant_err**2))/(1/(const_current_hall_constant_err**2)+1/(const_field_hall_constant_err**2))
final_hall_const_err = 1/np.sqrt(1/(const_current_hall_constant_err**2)+1/(const_field_hall_constant_err**2))
# print(f"finall hall constant: {final_hall_const}, error: {final_hall_const_err}")
mobility = final_hall_const*conductivity
mobility_err = np.sqrt((final_hall_const*conductivity_err)**2+(final_hall_const_err*conductivity)**2)
print(f"finall mobility: {mobility}, error: {mobility_err}")
density = 1/(final_hall_const*ELECTRON_CHARGE)
density_err = final_hall_const_err/(ELECTRON_CHARGE*final_hall_const**2)
print(f"density: {density}, error: {density_err}")

# PlotTemperaturesConcentration(cooling_measurments, heating_measurments, temp_magnetic_field, temp_current_sample, MagneticError(temp_magnetic_field), CurrentError(temp_current_sample))
# PlotTemperaturesMobility(cooling_measurments, heating_measurments, temp_magnetic_field, MagneticError(temp_magnetic_field))

start = 7
used_exp = -3/2
# print(f"Temperature measurment field setting: {temp_magnetic_field}, error: {MagneticError(temp_magnetic_field)}")
# print(f"Temperature measurment current: {temp_current_sample}, error: {CurrentError(temp_current_sample)}")
# print(TempError(cooling_measurments[0]))
# print(MultimeterVoltageError(cooling_measurments[1], HALL_RANGE))
# print(SampleVoltageError(cooling_measurments[2]))
# print(TempError(heating_measurments[0]))
# print(MultimeterVoltageError(heating_measurments[1], HALL_RANGE))
# print(SampleVoltageError(heating_measurments[2]))
# PlotTemperatures(cooling_measurments, heating_measurments)
# PlotTemperaturesVoltage(cooling_measurments, heating_measurments)
# PlotTemperaturesConductivity(cooling_measurments, heating_measurments)
# PlotTemperaturesConductivityAdjusted(cooling_measurments, heating_measurments)
# PlotTemperaturesConductivityAdjusted(cooling_measurments[:, start:], heating_measurments[:, start:])
conductivity_cooling, conductivity_cooling_err = Conductivity(cooling_measurments[2], temp_current_sample, SampleVoltageError(cooling_measurments[2]), CurrentError(temp_current_sample))
conductivity_heating, conductivity_heating_err = Conductivity(heating_measurments[2], temp_current_sample, SampleVoltageError(heating_measurments[2]), CurrentError(temp_current_sample))
conductivity_cooling_adj = np.log(conductivity_cooling[start:])
conductivity_cooling_adj_err = conductivity_cooling_err[start:]/conductivity_cooling[start:]
conductivity_heating_adj = np.log(conductivity_heating[start:])
conductivity_heating_adj_err = conductivity_heating_err[start:]/conductivity_heating[start:]
temp_cooling_kelvin = cooling_measurments[0, start:]+KELVINS
temp_heating_kelvin = heating_measurments[0, start:]+KELVINS
temp_cooling_adj = 1/temp_cooling_kelvin
temp_cooling_adj_err = TempError(temp_cooling_kelvin)/(temp_cooling_kelvin**2)
temp_heating_adj = 1/temp_heating_kelvin
temp_heating_adj_err = TempError(temp_heating_kelvin)/(temp_heating_kelvin**2)
cooling_fit, cooling_fit_err = FitToLinearModel(temp_cooling_adj, conductivity_cooling_adj, temp_cooling_adj_err, conductivity_cooling_adj_err)
heating_fit, heating_fit_err = FitToLinearModel(temp_heating_adj, conductivity_heating_adj, temp_heating_adj_err, conductivity_heating_adj_err)
# print(f"start temp: {cooling_measurments[0, start]}")
# print(f"start temp: {heating_measurments[0, start]}")
# print(f"parameters fited for cooling: {cooling_fit}, error: {cooling_fit_err}")
# print(f"parameters fited for heating: {heating_fit}, error: {heating_fit_err}")
# PlotLineFit(temp_cooling_adj, conductivity_cooling_adj, temp_cooling_adj_err, conductivity_cooling_adj_err, cooling_fit, r"$T^{-1} \ [K^{-1}]$", r"$\log(\sigma)$", "Punkty pomiarowe dla chłodzenia", "Dopasowanie Liniowe", "")
# PlotLineFit(temp_heating_adj, conductivity_heating_adj, temp_heating_adj_err, conductivity_heating_adj_err, heating_fit, r"$T^{-1} \ [K^{-1}]$", r"$\log(\sigma)$", "Punkty pomiarowe dla grzania", "Dopasowanie Liniowe", "")
coefficient = (cooling_fit[0]/(cooling_fit_err[0]**2)+heating_fit[0]/(heating_fit_err[0]**2))/(1/(cooling_fit_err[0]**2)+1/(heating_fit_err[0]**2))
coefficient_err = 1/np.sqrt(1/(cooling_fit_err[0]**2)+1/(heating_fit_err[0]**2))
# print(f"final value: {coefficient}, error: {coefficient_err}")
final_energy = coefficient*BOLTZMAN_CONST*2
final_energy_err = coefficient_err*BOLTZMAN_CONST*2
# print(f"final energy value: {final_energy/ELECTRON_CHARGE}, error: {final_energy_err/ELECTRON_CHARGE}")
# PlotTemperaturesExp(cooling_measurments, heating_measurments, used_exp)
# PlotTemperaturesLog(cooling_measurments, heating_measurments)
# PlotTemperaturesExp(cooling_measurments[:, start:], heating_measurments[:, start:])
cooling_measurments_line = cooling_measurments[0, start:]
heating_measurments_line = heating_measurments[0, start:]
data_cooling, data_cooling_err = TempMeasurmentsExp(cooling_measurments_line, used_exp);
data_heating, data_heating_err = TempMeasurmentsExp(heating_measurments_line, used_exp);
# cooling_fit_params, cooling_fit_params_err = FitToLinearModel(data_cooling, cooling_measurments[1, start:], data_cooling_err, MultimeterVoltageError(cooling_measurments[1, start:], HALL_RANGE))
# heating_fit_params, heating_fit_params_err = FitToLinearModel(data_heating, heating_measurments[1, start:], data_heating_err, MultimeterVoltageError(heating_measurments[1, start:], HALL_RANGE))

# print(f"parameters fited for cooling for temp range: [{min(cooling_measurments_line)};{max(cooling_measurments_line)}]")
# print(f"parameters fited for cooling: {cooling_fit_params}, error: {cooling_fit_params_err}")
# PlotLineFit(data_cooling/MILI, 
#             cooling_measurments[1, start:]/MILI, 
#             data_cooling_err/MILI, 
#             MultimeterVoltageError(cooling_measurments[1, start:], HALL_RANGE)/MILI, 
#             (cooling_fit_params[0], cooling_fit_params[1]/MILI), 
#             r"$T^{-\frac{3}{2}} \ [m \, ^{\circ^{-\frac{3}{2}}}]$", 
#             r"$U_{H} \ [mV]$", 
#             "punkty pomiarowe", 
#             "dopasowanie liniowe", 
#             r"Zależność napięcia halla od temperatury do potęgi $-\frac{3}{2}$" + f"\n dla temperatur w zakresie [{min(cooling_measurments_line)};{max(cooling_measurments_line)}] podczas chłodenia.")
# 
# print(f"parameters fited for heating for temp range: [{min(heating_measurments_line)};{max(heating_measurments_line)}]")
# print(f"parameters fited for heating: {heating_fit_params}, error: {heating_fit_params_err}")
# PlotLineFit(data_heating/MILI, 
#             heating_measurments[1, start:]/MILI, 
#             data_heating_err/MILI, 
#             MultimeterVoltageError(heating_measurments[1, start:], HALL_RANGE)/MILI, 
#             (heating_fit_params[0], heating_fit_params[1]/MILI), 
#             r"$T^{-\frac{3}{2}} \ [m \, ^{\circ^{-\frac{3}{2}}}]$", 
#             r"$U_{H} \ [mV]$", 
#             "punkty pomiarowe", 
#             "dopasowanie liniowe", 
#             r"Zależność napięcia halla od temperatury do potęgi $-\frac{3}{2}$" + f"\n dla temperatur w zakresie [{min(heating_measurments_line)};{max(heating_measurments_line)}] podczas grzania.")

