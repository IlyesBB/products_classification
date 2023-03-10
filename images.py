import cv2 as cv
import numpy as np
from PIL import ImageOps, Image


def get_new_shape(shape, max_size):
    """Calcule les nouvelles dimensions d'une image, en limitant la dimension max et en convervant les proportions"""
    # max_size_act: Dimension maximale actuelle de l'image
    max_size_act = max(shape)
    if max_size_act <= max_size:  # Si les dimensions sont suffisamment petites, on ne fait rien
        return shape
    # l_shape: Dimensions de l'image sous forme de liste
    l_shape = list(shape)
    # ind_max: Indice de la dimension maximale
    ind_max = l_shape.index(max_size_act)
    # min_over_max: Ratio entre la petite et la grande dimension
    min_over_max = l_shape[ind_max-1]/l_shape[ind_max]
    # On modifie les dimension, en préservant le rapport de proportionnalité
    l_shape[ind_max] = max_size
    l_shape[ind_max-1] = int(round(min_over_max*max_size))
    l_shape.reverse()  # On l'inverse pour la fonction resize
    return tuple(l_shape)
    
def correct_image(file_path, max_size=256, gray_shades=True):
    """Corrige le contraste, la luminosité et le bruit. Diminue la résolution de l'image si besoin."""
    img = cv.imread(file_path)
    if gray_shades:
        img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    else:
        img_gray = img
    pic_gray = ImageOps.autocontrast(Image.fromarray(img_gray))
    pic_gray = ImageOps.equalize(pic_gray)
    img_gray = cv.fastNlMeansDenoising(np.array(pic_gray))
    if not gray_shades:
        new_size = get_new_shape(img_gray.shape[:-1], max_size)
    else:
        new_size = get_new_shape(img_gray.shape, max_size)
    pic_gray = Image.fromarray(img_gray)
    if not gray_shades:
        if new_size != img_gray.shape[:-1]:
            pic_gray = pic_gray.resize(new_size)
    else:
        if new_size != img_gray.shape:
            pic_gray = pic_gray.resize(new_size)
    return pic_gray

def save_corrected_image(file_name, in_dir, out_dir, max_size=256, gray_shades=True):
    """Corrige l'image avec correct_image et l'enregistre dans un répertoire"""
    pic_gray = correct_image(in_dir+file_name, max_size, gray_shades)
    pic_gray.save(out_dir+file_name)
    
def get_descriptors(image_path):
    """Retourne les descripteur d'une image calculés par Sift"""
    img = cv.imread(image_path)
    sift = cv.SIFT_create()
    kp, des = sift.detectAndCompute(img, None)
    if des is None:
        return []
    else:
        return des

def correct_descriptor(desc):
    """Transforme un descripteur pour le rendre invariable par modification de l'exposition / du constraste"""
    # Invariance par modification du contraste:
    desc = desc/np.linalg.norm(desc)
    # Invariance par modification de l'exposition
    desc = np.where(desc>0.2, 0.2, desc)
    desc = desc/np.linalg.norm(desc)
    return desc

def add_descriptors_to_vocab(image_path, vocab):
    """Calcule les descripteurs, les modifie pour les rendre invariants par contraste et par exposition,
    et les enregistre dans un ensemble de descripteurs"""
    # descriptors: Liste des descripteurs de l'image
    descriptors = get_descriptors(image_path)    
    # desc: Un descripteur de l'image
    for desc in descriptors:
        desc = correct_descriptor(desc)
        # On ajoute le descripteur au vocabulaire d'images
        vocab.add(tuple(desc))