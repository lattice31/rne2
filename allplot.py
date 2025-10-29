import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la
import pandas as pd
kB = 8.617333262e-5  * 0.0734985857     # Boltzmann constant, Ry/K
#  * 0.0734985857 

def thermal_density_matrix(H, beta):
    expm = la.expm(-beta * H)
    Z = np.trace(expm)
    return expm / Z, Z

def excitation_spectrum_cm(energies_cm, T):
    # Convert to array and shift to zero minimum, then normalize
    if T <= 0:
        T = 1e-6
    energies = np.array(energies_cm)
    energies -= np.min(energies)
    energies = energies/4
    beta = 1.0 / (kB * T)
    H = np.diag(energies)
    rho, Z = thermal_density_matrix(H, beta)
    pops = np.real(np.diag(rho))
    mag = [1.0, 0.0, 0.0, 0.0]
    mag  = np.dot(pops, mag)   # Magnetic moment assumed proportional to ground state population
    return mag

if __name__ == "__main__":
    # Ph7 = [-2367.75317359, -2367.75194090, -2367.74835994, -2367.74921508]
    # Ph24 = [-2367.74490266, -2367.74422148, -2367.74008495, -2367.74054833]
    # Ph56 = [-2367.71410587, -2367.71271859, -2367.70589023, -2367.70699989]
    # relaxed = [-2367.75766477, -2367.75659546, -2367.75274822, -2367.75348119]

    # T_range = np.arange(1, 100.1, 0.1)

    # mag_ph7 = [excitation_spectrum_cm(Ph7, T) for T in T_range]
    # mag_ph24 = [excitation_spectrum_cm(Ph24, T) for T in T_range]
    # mag_ph56 = [excitation_spectrum_cm(Ph56, T) for T in T_range]
    # mag_relaxed = [excitation_spectrum_cm(relaxed, T) for T in T_range]
    df = pd.read_csv("energy.csv")
    # 에너지 열만 추출
    energy_cols = ["fm","afm2", "afm3", "afm"]
    ph_data = []
    labels = []

    for i, row in df.iterrows():
        energies = [row[c] for c in energy_cols]
        if len(energies) == 4:
            ph_data.append(energies)
            labels.append(row["folder"])

    # === 5개씩 묶기 ===
    group_size = 5
    grouped_data = [
        ph_data[i:i + group_size] for i in range(0, len(ph_data), group_size)
    ]
    grouped_labels = [
        labels[i:i + group_size] for i in range(0, len(labels), group_size)
    ]

    # === 온도 범위 ===
    T_range = np.arange(1, 100.1, 0.1)

    def add_label(x, y, text, color):
        plt.text(x, y, text, fontsize=8, color=color, fontweight='bold',
                 va='center', ha='left', backgroundcolor='white')

    # === 그래프 그리기 ===
    for idx, group in enumerate(grouped_data):
        plt.figure(figsize=(8, 6))

        for ph, label in zip(group, grouped_labels[idx]):
            mag_values = [excitation_spectrum_cm(ph, T) for T in T_range]
            plt.plot(T_range, mag_values, label=label)
            add_label(T_range[-1] + 1, mag_values[-1], label, "royalblue")

        plt.title(f"Magnetization for PH group {grouped_labels[idx][0]}–{grouped_labels[idx][-1]}", fontsize=14)
        plt.xlabel("Temperature (K)", fontsize=12)
        plt.ylabel("Magnetization", fontsize=12)
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.show()

    # plt.figure(figsize=(8,6))
    # plt.plot(T_range, mag_ph7, label="Ph7", color="royalblue")
    # plt.plot(T_range, mag_ph24, label="Ph24", color="seagreen")
    # plt.plot(T_range, mag_ph56, label="Ph56", color="darkorange")
    # plt.plot(T_range, mag_relaxed, label="relaxed", color="crimson")


    # # Find label positions near the right edge of plot
    # 
    # add_label(T_range[-1] + 1, mag_ph24[-1], "Ph24", "seagreen")
    # add_label(T_range[-1] + 1, mag_ph56[-1], "Ph56", "darkorange")
    # add_label(T_range[-1] + 1, mag_relaxed[-1], "relaxed", "crimson")
    # plt.title("Magnetization of main phonon modes", fontsize=14)
    # plt.xlabel("Temperature (K)", fontsize=12)
    # plt.ylabel("Magnetization", fontsize=12)
    # plt.legend()
    # plt.grid(True, linestyle="--", alpha=0.5)
    # plt.tight_layout()
    # plt.show()
