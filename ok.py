# Convertir les données en une matrice 2D
all_speeds = np.array(list(speed_m_per_sec_par_trajet.values()))

# Créer une figure pour le graphique 2D
fig, ax = plt.subplots(figsize=(10, 6))

vmin = all_speeds.min()
vmax = all_speeds.max()
vmax = 1
norm = colors.Normalize(vmin=vmin, vmax=vmax)


# Appliquer pcolormesh pour afficher la heatmap des vitesses
c = ax.pcolormesh(all_speeds, cmap='viridis', norm=norm,shading='auto')  # Utilisation de 'viridis' comme colormap

# Ajouter une barre de couleurs (colorbar)
fig.colorbar(c, ax=ax, label='Vitesse (m/s)')

# Ajouter un titre et des labels
ax.set_title("Heatmap des vitesses par trajet (échelle linéaire sur l'axe des y)")
ax.set_xlabel("Index de la vitesse")
ax.set_ylabel("Trajet")



filename = f'Vitesses_vs_time_{self.line_number}_{self.date_video}_{self.window_size_seconds}_{debut_echantillonnage:03}.png'
filepath = self.output_path /  filename
plt.savefig(filepath,dpi=self.DPI_SAVED_IMAGES)
