import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt

def index_wavelength_complex(wave_um, A, B):
    """
    wave_um: wavelength in micrometers (um)
    Returns: Refractive index via a single-term Sellmeier-like equation
             n = sqrt(1 + (A * lambda^2) / (lambda^2 - B))
    """
    return np.sqrt(1 + (wave_um**2 * A) / (wave_um**2 - B))

# -------------------------------------
# Original data (wavelengths in meters)
# -------------------------------------
wavelengths_m = np.array([447.1477e-9,
                          501.5618e-9,
                          587.5618e-9,
                          667.8151e-9])

# Convert to micrometers
wavelengths_um = wavelengths_m * 1e6

# Refractive index measurements
index_szklo = np.array([1.71433, 1.70132, 1.68892, 1.68159])

# Measurement errors
index_szklo_err = np.array([0.00018 for _ in range(4)])

# -------------------------------------
# Fit parameters
# -------------------------------------
# Better initial guess. For many glasses, "A" is often O(1), and "B" is 
# on the order of 0.01–0.1 (it depends on the glass type).
initial_guess = [1.0, 0.01]

# You can also set bounds if you have physical constraints
# E.g., (A, B) > 0 and not unreasonably large:
bounds = ((0, 0), (10, 10))

param_com, cov_mat_com = opt.curve_fit(
    index_wavelength_complex,
    wavelengths_um,
    index_szklo,
    p0=initial_guess,
    sigma=index_szklo_err,
    bounds=bounds
)

A_fit, B_fit = param_com
print("Fitted parameters:")
print(f"A = {A_fit:.6f}")
print(f"B = {B_fit:.6f}")

# -------------------------------------
# Evaluate the model on a fine grid
# -------------------------------------
x_model_um = np.linspace(min(wavelengths_um), max(wavelengths_um), 200)
y_model = index_wavelength_complex(x_model_um, A_fit, B_fit)

# -------------------------------------
# Plot data vs. fitted model
# -------------------------------------
plt.errorbar(
    wavelengths_um, index_szklo, yerr=index_szklo_err, fmt='o',
    label='Experimental data'
)
plt.plot(x_model_um, y_model, '-', label='Fitted curve')
plt.xlabel('Wavelength (µm)')
plt.ylabel('Refractive index')
plt.legend()
plt.show()
