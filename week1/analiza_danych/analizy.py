import numpy as np
import pandas as pd
import matplotlib as mp
import scipy
import matplotlib.pyplot as plt

MEASURE_ERR = 30/3600

def error_stat(arr):
    arr_len = len(arr)
    error_val = np.sqrt(np.sum((arr-np.mean(arr))**2) / (arr_len*(arr_len-1)))
    return error_val

def full_error(stat_error):
    return np.sqrt(stat_error**2 + (MEASURE_ERR**2)/2)

def sec_min(val):
    ret = []
    for _ in range(2):
        ret.append(int(val))
        val = (val - int(val))*60
    ret.append(val)
    return ret

def deg_frac(val):
    return val[0]+val[1]/60+val[2]/3600

def deg_to_rad(val):
    return val*np.pi/180

def rad_to_deg(val):
    return val*180/np.pi

def index_of_refraction(delta, alpha):
    up = np.sin(0.5*(delta+alpha))
    down = np.sin(0.5*alpha)
    return up/down

def index_of_refraction_error(delta, alpha, delta_err, alpha_err):
    del_dif = (np.cos(1/2*(delta+alpha)))/(2*np.sin(alpha/2))*delta_err
    alp_dif = (np.sin(1/2*delta))/(2*((np.sin(alpha/2))**2))*alpha_err
    return np.sqrt(del_dif**2 + alp_dif**2)

def full_analysis(arr):
    stat_error = error_stat(arr)
    err = full_error(stat_error)
    val = np.mean(arr)
    return [val, stat_error, err]

def full_index(phi, delta):
    phi_full = full_analysis(phi)
    delta_full = full_analysis(delta)
    phi_mean = deg_to_rad(phi_full[0])
    phi_error = deg_to_rad(phi_full[2])
    delta_mean = deg_to_rad(delta_full[0])
    delta_error = deg_to_rad(delta_full[2])
    index = index_of_refraction(delta_mean, phi_mean)
    index_err = index_of_refraction_error(delta_mean, phi_mean, delta_error, phi_error)
    return [[sec_min(i) for i in delta_full], index, index_err]

lamiacy_szklo = np.array([30, 0, 60, 90, 60, 90])/3600 + 45
lamiacy_kalcyt = np.array([8*60, 8*60+30, 7*60+30, 9*60])/3600+60
odchylenie_kalcyt_zwy = np.array([6*60, 6*60+30, 6*60+30])/3600+36
odchylenie_kalcyt_nadzwy = np.array([14*60, 13*60+30, 14*60])/3600+52
odchylenie_szklo_blue = np.array([30, 90, 30, 60])/3600 + 37
odchylenie_szklo_green = np.array([15*60, 15*60+30, 16*60, 16*60])/3600 + 36
odchylenie_szklo_yellow = np.array([33*60, 33*60, 33*60, 32*60])/3600 + 35
odchylenie_szklo_red = np.array([7*60+30, 7*60+30, 7*60, 8*60])/3600 + 35

print(full_index(lamiacy_kalcyt, odchylenie_kalcyt_zwy))
# --------------------------
# Define the two model forms
# --------------------------
def index_wavelength_simple(wave, a, b):
    """
    wave in meters
    Returns a + b/(wave^2)
    """
    return a + b / (wave**2)

def index_wavelength_complex(wave_um, A, B):
    """
    wave_um in micrometers
    Returns sqrt(1 + (A * wave_um^2) / (wave_um^2 - B))
    """
    return np.sqrt(1 + (wave_um**2 * A) / (wave_um**2 - B))

# --------------------------
# Input data
# --------------------------
wavelengths_m = np.array([447.1477e-9, 
                          501.5618e-9, 
                          587.5618e-9, 
                          667.8151e-9])
index_szklo = np.array([1.71433, 1.70132, 1.68892, 1.68159])
index_szklo_err = 5/4*np.array([0.00018, 0.00018, 0.00018, 0.00018])

# --------------------------
# Convert meters -> micrometers for the complex model
# --------------------------
wavelengths_um = wavelengths_m * 1e6

# --------------------------
# Fit the simple model (wave in meters)
# --------------------------
param_sim, cov_mat_sim = scipy.optimize.curve_fit(
    index_wavelength_simple,
    wavelengths_m,
    index_szklo,
    sigma=index_szklo_err
)

# --------------------------
# Fit the complex model (wave in micrometers)
# Provide a reasonable p0 and also pass sigma
# --------------------------
param_com, cov_mat_com = scipy.optimize.curve_fit(
    index_wavelength_complex,
    wavelengths_um,
    index_szklo,
    p0=[1.0, 0.01],
    sigma=index_szklo_err
)

# Print fitted parameters
print("Simple model fit parameters:")
print(f"  a = {param_sim[0]:.18f}")
print(f"  b = {param_sim[1]:.18f}")
print("\nCovariance matrix (simple):")
print(cov_mat_sim)

print("\nComplex model fit parameters:")
print(f"  A = {param_com[0]:.16f}")
print(f"  B = {param_com[1]:.16f}")
print("\nCovariance matrix (complex):")
print(cov_mat_com)

# --------------------------
# Compute chi^2 and reduced chi^2 for both models
# --------------------------
# Simple model predictions (use the data's wavelengths in meters)
y_pred_sim = index_wavelength_simple(wavelengths_m, *param_sim)
chi2_sim = np.sum(((index_szklo - y_pred_sim) / index_szklo_err)**2)
dof_sim = len(index_szklo) - len(param_sim)  # degrees of freedom
p_simple = scipy.stats.chi2.sf(chi2_sim, dof_sim)
red_chi2_sim = chi2_sim / dof_sim

# Complex model predictions (use the data's wavelengths in micrometers)
y_pred_com = index_wavelength_complex(wavelengths_um, *param_com)
chi2_com = np.sum(((index_szklo - y_pred_com) / index_szklo_err)**2)
dof_com = len(index_szklo) - len(param_com)
p_complex = scipy.stats.chi2.sf(chi2_com, dof_com)
red_chi2_com = chi2_com / dof_com

print("=== Chi-squared Statistics ===")
print(f"Simple Model:  chi^2 = {chi2_sim:.4f},  dof = {dof_sim},  reduced chi^2 = {red_chi2_sim:.4f}, probability = {p_simple:.4f}")
print(f"Complex Model: chi^2 = {chi2_com:.4f}, dof = {dof_com}, reduced chi^2 = {red_chi2_com:.4f}, probability = {p_complex:.4f}\n")

# --------------------------
# Build a fine grid for plotting both models
# --------------------------
x_model_m = np.linspace(wavelengths_m.min(), wavelengths_m.max(), 200)
# For the complex model, convert that same grid to micrometers:
x_model_um = x_model_m * 1e6

y_model_sim = index_wavelength_simple(x_model_m, param_sim[0], param_sim[1])
y_model_com = index_wavelength_complex(x_model_um, param_com[0], param_com[1])

# --------------------------
# Plot
# --------------------------
plt.errorbar(wavelengths_m, index_szklo, yerr=index_szklo_err, fmt='o', label='Punkty Pomiarowe')
plt.plot(x_model_m, y_model_sim, '-', label='Model Cauchego')
plt.plot(x_model_m, y_model_com, '-', label='Model Sellmeier')
plt.xlabel('Długość fali [m]')
plt.ylabel('Współczynnik załamania')
plt.title('Zależność współczynnika załamania od długości fali')
plt.legend()
plt.show()
