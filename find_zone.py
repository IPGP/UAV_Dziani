import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt



#Dimensions réelles en métres de la barge
Largeur_barge_en_m = 2
hauteur_barge_en_m = 2

bleu  = (255,0,0)
black = (0,0,0)
white = (255,255,255)

red = (255,0,0)
green = (0,255,0)
blue = (0,0,255)

yellow = (255,255,0)
magenta = (255,0,255)
cyan = (0,255,255)

coef_resize = 2

def detect_objects( frame,threshold_area_contours):
    #  Convert Image to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    C = 2
    pixel_neighborhood_size = 15
    # Create a Mask with adaptive threshold
    mask = cv2.adaptiveThreshold(gray, 255,
                                 cv2.ADAPTIVE_THRESH_MEAN_C,
                                 cv2.THRESH_BINARY_INV,
                                 pixel_neighborhood_size, C)

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    #cv2.imshow("mask", mask)
    objects_contours = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > threshold_area_contours:
            #cnt = cv2.approxPolyDP(cnt, 0.03*cv2.arcLength(cnt, True), True)
            objects_contours.append(cnt)

    return objects_contours

def is_rectangular(contour, epsilon=0.02):
    """Approximate contour to a polygon and check if it is rectangular."""
    approx = cv2.approxPolyDP(contour, epsilon * cv2.arcLength(contour, True), True)
    if len(approx) == 4:
        return True
    return False




def trace_mask(video_path):

    video_file = cv2.VideoCapture(video_path)
    if not video_file.isOpened():
        print(f"Error while opening video file {video_path}")
        sys.exit()


    # Lecture de la première frame et création
    frame_available, image = video_file.read()
    image_ori=image.copy()



    #cv2.imshow('image_ori', cv2.resize(image_ori,
    #                                   (int(image_ori.shape[1]/coef_resize),
    #                                     int(image_ori.shape[0]/coef_resize))))
    #cv2.waitKey(0)



    if not frame_available:
        print(f"Erreur de lecture de la première frame de {video_path}")
        video_file.release()
        sys.exit()

    # 15000 permet a priori de ne trouver que le contour de la grande mousse
    threshold_area_contours = 15000
    threshold_area_contours = 5000
    contours = detect_objects(image,threshold_area_contours)

    # Au cas ou, on prend le contour avec la plus grande surface
    big_contour = max(contours, key=cv2.contourArea)


    # Original contour en rouge
    cv2.polylines(image, [big_contour], True, red,2)

    # smooth contour
    #peri = cv2.arcLength(big_contour, True)
    #big_contour_smoothed = cv2.approxPolyDP(big_contour, 0.0008 * peri, True)

    #cv2.polylines(image, [big_contour_smoothed], True, blue,2)

    mask_mousse = np.zeros(image.shape, np.uint8)

    #cv2.drawContours(mask_mousse, contours, -1, (255,255,255),1)
    #cv2.drawContours(mask_mousse, contours,-1, 255, cv2.FILLED, 1)
    cv2.drawContours(mask_mousse, contours, -1, white, cv2.FILLED, 1)


    out=image_ori.copy()
    out[mask_mousse == 0] = 0


    numpy_horizontal = np.hstack((cv2.resize(image_ori,
                                    (int(image_ori.shape[1]/coef_resize),
                                        int(image_ori.shape[0]/coef_resize))),
                                        cv2.resize(out, (int(out.shape[1]/coef_resize),
                                                            int(out.shape[0]/coef_resize)))))


    #cv2.imshow('out', cv2.resize(out, (int(out.shape[1]/coef_resize), int(out.shape[0]/coef_resize))))
    if '/' in video_path:
        out_name =video_path.split('/')[-1]
        out_name = out_name.split('.')[0]
    else :
        out_name = out_name.split('.')[0]

    cv2.imwrite(f'{out_name}_mask.png',numpy_horizontal)
#    cv2.imshow('_', numpy_horizontal)
 #   cv2.waitKey(0)




# Pas ok

videos= []
videos.append('./25_11_2022_DJI_0007.MOV')
videos.append('./24_11_2022_DJI_0002.MOV')

# Ok
videos.append( './2024_04_mayotte/24_04_08/DJI_0096.MOV' )
videos.append('./DJI_0033.MOV' )

for video in videos:
    trace_mask(video)

sys.exit()



# 2. Conversion en niveaux de gris
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Display the gray_image
cv2.imshow('Binary Mask', cv2.resize(gray_image, (1280,720)))
cv2.waitKey(0)

# Apply thresholding to segment the black object
_, mask = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)

# Détection des régions connectées
nb_regions, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

# Paramètres de filtrage
seuil_nb_pixels = 5000  # Seuil du nombre de pixels pour sélectionner les régions

# Filtrage des régions
regions_selectionnees = []
for region_id in range(1, nb_regions):
    nb_pixels = stats[region_id, cv2.CC_STAT_AREA]
    if nb_pixels > seuil_nb_pixels:
        regions_selectionnees.append(region_id)

# Création d'une nouvelle image avec les régions sélectionnées
image_filtree = np.zeros_like(mask)
for region_id in regions_selectionnees:
    image_filtree[labels == region_id] = 255

# Afficher l'image filtrée
cv2.imshow("zone_bullage_" , image_filtree)
cv2.waitKey(0)


sys.exit()
# Display the binary mask
cv2.imshow('Binary Mask', cv2.resize(binary_mask, (1280,720)))
cv2.waitKey(0)

# Find contours in the binary image
contours, _ = cv2.findContours(binary_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

# Sort contours by area in descending order
contours = sorted(contours, key=cv2.contourArea, reverse=True)

# Exclude the largest contour (if there are more than one contour)
if len(contours) > 1:
    contours = contours[1:]

# Create an image for visualizing contours
contours_image = np.zeros_like(image)

# Draw contours on the visualization image
cv2.drawContours(contours_image, contours, -1, (0, 225, 0), 2)

# Display the image with contours
cv2.imshow('Contours', cv2.resize(contours_image, (1280,720)))
cv2.waitKey(0)

# Create an empty image to draw the convex hull
hull_image = np.zeros_like(image)

# Fusionner tous les contours en un seul
merged_contour = np.vstack(contours)

# Obtenir l'enveloppe convexe du contour fusionné
hull = cv2.convexHull(merged_contour)

# Trouver le rectangle incliné englobant
rect = cv2.minAreaRect(hull)

# Dessiner le rectangle incliné sur l'image originale
box = cv2.boxPoints(rect)
box = np.int0(box)
cv2.drawContours(image, [box], 0, (0, 0, 255), 2)

# Affichage des dimensions du rectangle
largeur_barge_en_px, hauteur_barge_en_px = rect[1]
print(f"Largeur en pixels: {largeur_barge_en_px}, Hauteur en pixels: {hauteur_barge_en_px}")

# Display the result
cv2.imshow('Bounding Rectangle', cv2.resize(image, (1280,720)))
cv2.waitKey(0)
cv2.destroyAllWindows()


sys.exit()
# 3. Détection des bords
# Ajuster les seuils de Canny pour une sensibilité accrue
edges = cv2.Canny(gray, 10, 70)  # Réduire les seuils pour détecter plus de bords

# 4. Masque de mousse blanche
# Supposons que la mousse blanche soit principalement blanche, nous pouvons créer un masque pour cette région.
# Ajuster les valeurs des seuils si nécessaire.
lower_white = np.array([200, 200, 200], dtype=np.uint8)
upper_white = np.array([255, 255, 255], dtype=np.uint8)
mask_white = cv2.inRange(image, lower_white, upper_white)

# Inverser le masque pour garder tout sauf la mousse blanche
mask = cv2.bitwise_not(mask_white)

# Appliquer le masque à l'image des bords
masked_edges = cv2.bitwise_and(edges, edges, mask=mask)

# 5. Détection de contours
contours, _ = cv2.findContours(masked_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 6. Filtrage des contours pour éliminer les formes rectangulaires
filtered_contours = []
epsilon = 0.01  # Réduire l'epsilon pour une approximation plus précise
for contour in contours:
    if not is_rectangular(contour, epsilon):
        filtered_contours.append(contour)

# Supposons que la zone de forme patatoïde est le plus grand contour non rectangulaire trouvé
filtered_contours = sorted(filtered_contours, key=cv2.contourArea, reverse=True)
if filtered_contours:
    patatoid_contour = filtered_contours[0]

    # 7. Affichage du résultat
    output_image = image.copy()
    cv2.drawContours(output_image, [patatoid_contour], -1, (0, 255, 0), 3)

    # Afficher les résultats
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.title('Image Originale')
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title('Masque de Mousse Blanche Inversé')
    plt.imshow(mask, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title('Contours Détectés')
    plt.imshow(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.show()
else:
    print("Aucun contour trouvé correspondant à la forme patatoïde.")
