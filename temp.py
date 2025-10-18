import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la

# Boltzmann constant in Ry/K
kB = 8.617333262e-5 * 0.0734985857


en = np.loadtxt('en.txt',usecols=(4))

print(en)

# Function to compute thermal density matrix
def thermal_density_matrix(H, beta):
    expm = la.expm(-beta * H)
    Z = np.trace(expm)
    return expm / Z, Z

# Function to compute magnetic moment at a given temperature
def excitation_spectrum_cm(energies_cm,mag, T):
    # Convert to array and shift to zero minimum, then normalize
    energies_J = np.array(energies_cm)
    beta = 1.0 / (kB * T)
    H = np.diag(energies_J)
    rho, Z = thermal_density_matrix(H, beta)
    pops = np.real(np.diag(rho))
    mag  = np.dot(pops, mag)   # Magnetic moment assumed proportional to ground state population
    return mag

# Energy levels in cm⁻¹ (example: two levels)
energies1 =  en 
#energies1=np.array([   -2367.74306026,   -2367.74200928,   -2367.73959512,   -2367.73871510    ])
mag1      = np.array([ 0.0, 0.0, 0.0,1.0 ])
energies1 = (energies1 - np.min(energies1)) / 4.0  # Normalize energy scale

# Temperature range: 1K to 100K in 10K steps
temperatures = np.arange(1, 101, 10)
magnetic_moments = [excitation_spectrum_cm(energies1,mag1, T) for T in temperatures]



# Example: assuming temperatures and magnetic_moments are lists or numpy arrays
# Combine them into two columns
data = np.column_stack((temperatures, magnetic_moments))

# Save to a .txt file with a header
np.savetxt('plot_mag.txt', data, header='Temperature\tMagneticMoment', fmt='%.6f', delimiter='\t')


# Plotting T vs Magnetic Moment
plt.figure(figsize=(8, 5))
plt.plot(temperatures, magnetic_moments, marker='o', linestyle='-', linewidth=2)
plt.title('Magnetic Moment vs Temperature', fontsize=14)
plt.xlabel('Temperature (K)', fontsize=12)
plt.ylabel('Magnetic Moment (arb. units)', fontsize=12)
plt.ylim(0,1.1)
plt.grid(True)
plt.tight_layout()
plt.show()

