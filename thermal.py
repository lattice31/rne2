import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la

# Physical constants
kB  = 8.617333262 *10**-5 * 0.0734985857        # Boltzmann constant, Ry/K
ry2cm1 = 1/ (0.00012*0.0734985857 )

def thermal_density_matrix(H, beta):
    """Compute ρ = exp(−βH)/Z and the partition function Z."""
    print(f"Computing thermal density matrix for H:\n{H}\nwith β = {beta:.4e}")
    expm = la.expm(-beta * H)
    print(f"Exponential matrix:\n{expm}")
    Z = np.trace(expm)
    return expm / Z, Z

def excitation_spectrum_cm(energies_cm, T):
    """
    energies_cm : array-like of energy levels in cm⁻¹
    T           : temperature in K

    Returns: populations (array), spectrum (list of (ΔE_cm, intensity))
    """
    # Convert to Joules: E[J] = (h·c) × (energy in cm⁻¹)
    energies_J = np.array(energies_cm)

    # Inverse temperature β = 1/(k_B T)
    beta = 1.0 / (kB * T)

    # Build diagonal Hamiltonian
    H = np.diag(energies_J)
    print(f"Hamiltonian matrix:\n{H}")

    # Thermal density matrix
    rho, Z = thermal_density_matrix(H, beta)

    # Boltzmann populations p_i = ⟨i|ρ|i⟩
    pops = np.real(np.diag(rho))

    print(f"occupation matrix:\n{rho}")
    # Build spectrum: transitions i→j with ΔE in cm⁻¹
    spectrum = []
    mag = 0.0 
    mag = 1* pops[0] +3.7/4. *pops[1]

    print(pops)
    print(mag)
    return pops, spectrum, Z


if __name__ == "__main__":
    # Example: six levels at 0, 500, 1000, 1500, 2000, 2500 cm⁻¹
    Ph7 =  [-2367.75317359, -2367.75194090, -2367.74835994, -2367.74921508]
    Ph24 =  [-2367.74490266, -2367.74422148, -2367.74008495, -2367.74054833],
    Ph56 = [-2367.71410587, -2367.71271859, -2367.70589023, -2367.70699989],
    relaxed = [-2367.75766477, -2367.75659546, -2367.75274822, -2367.75348119]
    T_range = np.arange(0.1, 100.1, 0.1)

    mag_ph7 = [excitation_spectrum_cm(Ph7, T) for T in T_range]
    mag_ph24 = [excitation_spectrum_cm(Ph24, T) for T in T_range]
    mag_ph56 = [excitation_spectrum_cm(Ph56, T) for T in T_range]
    mag_relaxed = [excitation_spectrum_cm(relaxed, T) for T in T_range]

    plt.title("Magnetization vs Temperature", fontsize=14)
    plt.xlabel("Temperature (K)", fontsize=12)
    plt.ylabel("Magnetization (arb. units)", fontsize=12)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()