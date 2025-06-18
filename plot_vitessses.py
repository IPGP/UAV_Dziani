import numpy as np

# Définir la valeur du seuil (threshold)
threshold = 0.8
#all_speeds = np.array(list(speed_m_per_sec_par_trajet.values()))
max_length = max(len(v) for v in speed_m_per_sec_par_trajet.values())
all_speeds = np.array([np.pad(v, (0, max_length - len(v)), constant_values=np.nan)    for v in speed_m_per_sec_par_trajet.values()])

# Créer une figure pour le graphique 2D
fig, ax = plt.subplots(figsize=(10, 6))

# Appliquer un colormap aux vitesses, mais dessiner les courbes en noir
for i in range(all_speeds.shape[0]):  # Parcourir chaque trajet
    speed_data = all_speeds[i]  # Vitesse pour ce trajet
    ax.plot(speed_data, color='black', lw=1)  # Courbes en noir

# Ajouter une ligne horizontale au niveau du seuil
ax.axhline(y=threshold, color='red', linestyle='--', label=f'Seuil = {threshold}')

# Ajouter des marqueurs étoiles pour les points au-dessus du seuil
for i in range(all_speeds.shape[0]):
    speed_data = all_speeds[i]
    above_threshold = speed_data > threshold
    ax.scatter(np.where(above_threshold)[0], speed_data[above_threshold], color='black', marker='*', s=100, label='Points au-dessus du seuil' if i == 0 else "")

# Ajouter un titre et des labels
ax.set_title(f"Vitesses par Trajet (Seuil = {threshold})")
ax.set_xlabel("Index de la vitesse")
ax.set_ylabel("Vitesse (m/s)")


# Ajouter la légende
ax.legend()

filename = f'Vitesses_vs_time_{self.line_number}_{self.date_video}_{self.window_size_seconds}_{debut_echantillonnage:03}.png'
filepath = self.output_path /  filename
plt.savefig(filepath,dpi=self.DPI_SAVED_IMAGES)
