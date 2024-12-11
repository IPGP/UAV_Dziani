
# Nested bars
from time import sleep
from matplotlib import colors, pyplot as plt
import numpy as np
from tqdm import trange




status_modulo = 1
colormap =  plt.cm.rainbow
movie_length_seconds = 600
window_size_seconds = 20
windows_shift_seconds = 5
frames_per_window = window_size_seconds * 20


#table_colors = cmap(np.linspace(0, 1,))
nb_shift_total =int(movie_length_seconds - window_size_seconds/windows_shift_seconds)
print(f'nb_shift_total {nb_shift_total}')

table_colors = plt.colormaps.get_cmap('plasma').resampled(nb_shift_total).colors
#table_colors = colors.rgb2hex(colormap(np.linspace(0, 1,int(movie_length_seconds- window_size_seconds/windows_shift_seconds))))


for nb_shift in range(nb_shift_total):
    for frame_count in trange(frames_per_window,desc=f'{nb_shift}  ',
                                    miniters=status_modulo,
                                    position=nb_shift,
                                    colour=colors.rgb2hex(table_colors[nb_shift])):

        3*5
        #sleep(0.001)
