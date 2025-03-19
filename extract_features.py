import librosa
import torch
import numpy as np
import matplotlib.pyplot as plt
from kymatio.torch import Scattering1D


def display_coeff(audio_path):
    """ """

    # Charger l'audio
    y, sr = librosa.load(audio_path, sr=None)  # sr=None pour garder le taux d'échantillonnage original

    # Normalisation
    y = y / np.max(np.abs(y))  # Normaliser entre -1 et 1

    # Afficher des informations
    print(f"[+] Fréquence d'échantillonnage : {sr} Hz")
    print(f"[+] Nombre d'échantillons : {len(y)}")

    # Définition des paramètres du Scattering Transform
    J = 6  # Nombre d'échelles (contrôle la résolution temps/fréquence)
    Q = 8  # Nombre de bandes de fréquences par octave

    # Définition du module de Scattering
    scattering = Scattering1D(J=J, shape=(len(y),), Q=Q)

    # Conversion en tenseur PyTorch
    y_torch = torch.from_numpy(y).float()

    # Application du Scattering Transform
    scattered_features = scattering(y_torch)

    # Affichage des dimensions du résultat
    print(f"[+] Dimensions des features extraites : {scattered_features.shape}")

    # plot figure
    plt.figure(figsize=(10, 5))
    plt.imshow(scattered_features.numpy(), aspect='auto', origin='lower', cmap='jet')
    plt.colorbar(label="Amplitude")
    plt.xlabel("Temps")
    plt.ylabel("Coefficients Scattering")
    plt.title("Représentation des coefficients du Scattering Transform")
    plt.show()



if __name__ == "__main__":

    # params
    audio_path = "data/H02_20230420_112000.ogg"
    
    # run
    display_coeff(audio_path)
