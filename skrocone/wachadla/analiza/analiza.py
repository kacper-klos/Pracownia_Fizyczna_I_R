import scipy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

plt.rcParams.update({"font.size": 14})

IMAGE_PATH = "images"
raw = pd.read_csv("analiza/Dane Capstone.csv", sep=';', decimal=',')
SAMPLE_FREQ = 40 #Hz
DT = 1/SAMPLE_FREQ
SPACE = 3000
GRAVITY = 9.81
MASS = 3000
MASS_ERR = 40
MASS_CENTRE = 79.7
MASS_CENTRE_ERR = 0.1
CM = 1e-2
KG = 1e3

spring_data = np.array([
    [49.14, 91.27, 151.85, 201.43, 250.71, 299.12, 348.18],
    [31.4, 32.8, 34.8, 36.6, 38.2, 39.7, 41.4]
    ])
DISTANCE_ERROR = 0.3

spring_height = np.array([30, 40, 45])
spring_height_err = 0.1


def SimpleModel(t, A, omega, phi, b):
    return A * np.cos(omega*t + phi) + b
def BeatModel(t, A, omega_d, omega_s, phi_d, phi_s, b):
    return A * np.cos(omega_d*t + phi_d) * np.cos(omega_s + phi_s) + b
def LineModel(x, a, b):
    return a*x + b

def PlotOscilations(title, first_series, second_series = None, legend=False):
    plt.figure(figsize=(16,4))
    plt.plot(first_series[0], first_series[1], label="$x_1$")
    if second_series is not None:
        plt.plot(second_series[0], second_series[1], label="$x_2$")

    plt.xlabel("t [s]")
    plt.ylabel("x [m]")
    plt.grid(True)
    if legend:
        plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGE_PATH, f"{title}.png"), dpi=300, bbox_inches="tight")
    plt.clf()


def ParseData(raw, index):
    time_col = raw.columns[index*3]
    a_col = raw.columns[index*3+1]
    b_col = raw.columns[index*3+2]

    time = raw[time_col]
    a = raw[a_col]
    b = raw[b_col]

    mask_a = a.notna()
    mask_b = b.notna()

    a_combined = [time[mask_a].to_numpy(), a[mask_a].to_numpy()]
    b_combined = [time[mask_b].to_numpy(), b[mask_b].to_numpy()]
    return a_combined, b_combined

def Interpolate(combined):
    t_uniform = np.arange(combined[0].min(), combined[0].max(), DT)
    interpolation = scipy.interpolate.interp1d(combined[0], combined[1], kind='linear', fill_value='extrapolate')
    val_uniform = interpolation(t_uniform)
    return [t_uniform, val_uniform]

def FourierTransform(interpolated_data):
    freq = np.fft.rfftfreq(len(interpolated_data[1]), d=DT)
    fft = np.abs(np.fft.rfft(interpolated_data[1] - interpolated_data[1].mean()))
    return [freq, fft]

def PlotFourier(title, first_series, second_series = None, legend=False):
    plt.figure(figsize=(16, 4))
    plt.plot(first_series[0], first_series[1], label="$F(x_1)$")
    if second_series:
        plt.plot(second_series[0], second_series[1], label="$F(x_2)$")
    plt.xlim(0, 2)
    plt.xlabel("f [Hz]")
    plt.ylabel("A [m]")
    plt.tight_layout()
    if legend:
        plt.legend()
    plt.savefig(os.path.join(IMAGE_PATH, f"{title}_fft.png"), dpi=300, bbox_inches="tight")
    plt.clf()

def FindFourierPeaks(fourier):
    threshold = fourier[1].max() * 0.2
    peaks, props = scipy.signal.find_peaks(fourier[1], height=threshold, distance = 3)
    peak_heights = props['peak_heights']
    top2_idx = np.argsort(peak_heights)[-2:]
    best_peaks = peaks[top2_idx]
    best_freqs = fourier[0][best_peaks]
    best_amp = fourier[1][best_peaks]
    return best_freqs, best_amp

def CurveFittingSimple(values, freq, phi_init = 0):
    A = (values[1].max() - values[1].min())/2
    b = values[1].mean()
    omega = 2*np.pi*freq[0]
    phi = phi_init
    p0 = [A, omega, phi, b]
    param, param_err = scipy.optimize.curve_fit(SimpleModel, values[0], values[1], p0=p0)
    return param, param_err

def CurveFittingBeats(values, freq, phi_init_d = 0, phi_init_s = 0):
    A = (values[1].max() - values[1].min())/2
    b = values[1].mean()
    omega_d = 2*np.pi*freq[0]
    omega_s = 2*np.pi*freq[1]
    p0 = [A, omega_d, omega_s, phi_init_d, phi_init_s, b]
    print(p0)
    param, param_err = scipy.optimize.curve_fit(BeatModel, values[0], values[1], p0=p0)
    return param, param_err

def SingleAnalysisSimple(index):
    title = f"counterphase_{index//2+1}"
    a_combined, b_combined = ParseData(raw, index)
    a_fourier = FourierTransform(Interpolate(a_combined))
    b_fourier = FourierTransform(Interpolate(b_combined))
    a_freq, a_amp = FindFourierPeaks(a_fourier)
    b_freq, b_amp = FindFourierPeaks(b_fourier)
    a_freq_err = SAMPLE_FREQ/(2*len(a_combined[0]))
    print(f"częstotliwość transformacji fouriera przeciwfazy A: {a_freq}, z błędem {a_freq_err}")
    print(f"częstotliwość transformacji fouriera przeciwfacy B: {b_freq}, z błędem {a_freq_err}")
    PlotOscilations(title, b_combined, a_combined, True)
    PlotFourier(title, a_fourier, b_fourier, True)
    return a_freq[0], a_freq_err

def SingleAnalysisBeats(index):
    title = f"beats_{index//2+1}"
    a_combined, b_combined = ParseData(raw, index)
    a_fourier = FourierTransform(Interpolate(a_combined))
    b_fourier = FourierTransform(Interpolate(b_combined))
    a_freq, a_amp = FindFourierPeaks(a_fourier)
    b_freq, b_amp = FindFourierPeaks(b_fourier)
    a_freq_err = SAMPLE_FREQ/(2*len(a_combined[0]))
    print(f"częstotliwość transformacji fouriera dudnienia A: {a_freq}, z błędem {a_freq_err}")
    print(f"częstotliwość transformacji fouriera dudnienia B: {b_freq}, z błędem {a_freq_err}")
    PlotFourier(title, a_fourier, b_fourier, True)
    PlotOscilations(title, b_combined, a_combined, True)
    return a_freq[0], a_freq[1] , a_freq_err

def MomentOfInercia(freq, freq_err):
    inercia = MASS*GRAVITY*MASS_CENTRE/((2*np.pi*freq)**2) * CM/KG
    inercia_err = inercia * np.sqrt((MASS_ERR/MASS)**2 + (MASS_CENTRE_ERR/MASS_CENTRE)**2 + (2*freq_err/freq)**2)
    print(f"moment inercji: {inercia}, z błędem {inercia_err}")
    return inercia, inercia_err


def SpringAnalysis():
    param, param_err = scipy.optimize.curve_fit(LineModel, spring_data[0], spring_data[1], sigma=DISTANCE_ERROR)
    param_err = np.sqrt(np.diag(param_err))
    print(f"spring fit: {param}, error: {param_err}")

    spring_const = GRAVITY/param[0]
    spring_const_err = GRAVITY/(param[0]**2)*param_err[0]
    print(f"spring constant: {spring_const}, error: {spring_const_err}")

    plt.errorbar(spring_data[0], spring_data[1], yerr=DISTANCE_ERROR, fmt='o')
    input_fit = np.linspace(spring_data[0].min(), spring_data[0].max(), SPACE)
    y_fit = LineModel(input_fit, *param)
    plt.plot(input_fit, y_fit)

    plt.xlabel("m [g]")
    plt.ylabel("x [m]")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGE_PATH, f"spring_mass.png"), dpi=300, bbox_inches="tight")
    plt.clf()

def NoSpringPendulum():
    a_combined, b_combined = ParseData(raw, 6)
    title = "no_spring"
    b_fourier = FourierTransform(Interpolate(b_combined))
    b_freq, b_amp = FindFourierPeaks(b_fourier)
    b_freq_err = SAMPLE_FREQ/(2*len(a_combined[0]))
    inercia, inercia_err = MomentOfInercia(b_freq, b_freq_err)
    print(f"częstotliwość transformacji fouriera: {b_freq}, z błędem {b_freq_err}")
    PlotFourier(title, b_fourier)
    PlotOscilations(title, b_combined)
    return b_freq, b_freq_err, inercia, inercia_err

def EigenFrequenciesSpringConstant(high_freq, high_freq_err, low_freq, low_freq_err, inercia, inercia_err, spring_distance, spring_distance_err):
    part = 2*np.pi**2*inercia/(spring_distance**2)
    spring_const = part*(high_freq**2-low_freq**2)
    spring_distance_dif = 2*spring_const*spring_distance_err/spring_distance
    inercia_dif = spring_const*inercia_err/inercia
    high_freq_dif = 2*part*high_freq*high_freq_err
    low_freq_dif = 2*part*low_freq*low_freq_err
    spring_const_err = np.sqrt((spring_distance_dif)**2 + (inercia_dif)**2 +(high_freq_dif)**2 +(low_freq_dif)**2)
    print(f"Stała sprężyny: {spring_const}, error: {spring_const_err}")
    return spring_const, spring_const_err

def WeightedAverage(val, val_err):
    inverse_square = 1/val_err**2
    inverse_sum = 1/(np.sum(inverse_square))
    vals_weighted = np.sum(inverse_square*val)*inverse_sum
    vals_weighted_err = np.sqrt(inverse_sum)
    print(f"Średnia ważona: {vals_weighted}, error: {vals_weighted_err}")
    return vals_weighted, vals_weighted_err

low_freq, low_freq_err, inercia, inercia_err = NoSpringPendulum()

beats_vals = [[], []]
counterphase_vals = [[], []]

high_freq, high_freq_err = SingleAnalysisSimple(1)
vals = EigenFrequenciesSpringConstant(high_freq, high_freq_err, low_freq, low_freq_err, inercia, inercia_err, spring_height[0]*CM, spring_height_err*CM)
counterphase_vals[0].append(vals[0])
counterphase_vals[1].append(vals[1])
beats = np.sort(SingleAnalysisBeats(0))
vals = EigenFrequenciesSpringConstant(beats[2], beats[0], beats[1], beats[0], *MomentOfInercia(beats[1], beats[0]), spring_height[0]*CM, spring_height_err*CM)
beats_vals[0].append(vals[0])
beats_vals[1].append(vals[1])

high_freq, high_freq_err = SingleAnalysisSimple(3)
vals = EigenFrequenciesSpringConstant(high_freq, high_freq_err, low_freq, low_freq_err, inercia, inercia_err, spring_height[1]*CM, spring_height_err*CM)
counterphase_vals[0].append(vals[0])
counterphase_vals[1].append(vals[1])
beats = np.sort(SingleAnalysisBeats(2))
vals = EigenFrequenciesSpringConstant(beats[2], beats[0], beats[1], beats[0], *MomentOfInercia(beats[1], beats[0]), spring_height[1]*CM, spring_height_err*CM)
beats_vals[0].append(vals[0])
beats_vals[1].append(vals[1])

high_freq, high_freq_err = SingleAnalysisSimple(5)
vals = EigenFrequenciesSpringConstant(high_freq, high_freq_err, low_freq, low_freq_err, inercia, inercia_err, spring_height[2]*CM, spring_height_err*CM)
counterphase_vals[0].append(vals[0])
counterphase_vals[1].append(vals[1])
beats = np.sort(SingleAnalysisBeats(4))
vals = EigenFrequenciesSpringConstant(beats[2], beats[0], beats[1], beats[0], *MomentOfInercia(beats[1], beats[0]), spring_height[2]*CM, spring_height_err*CM)
beats_vals[0].append(vals[0])
beats_vals[1].append(vals[1])

WeightedAverage(np.array(counterphase_vals[0]), np.array(counterphase_vals[1]))
WeightedAverage(np.array(beats_vals[0]), np.array(beats_vals[1]))

#SpringAnalysis()
