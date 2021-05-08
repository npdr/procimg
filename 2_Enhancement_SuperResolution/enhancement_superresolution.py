# Trabalho 2 - Realce e Superresolução
# SCC0251 - Image Processing (01/2021)
# Fabiana Dalacqua Mendes - 9894536
# Pedro Henrique Nieuwenhoff - 10377729
import numpy as np
import imageio
import math

def load_images(name_base):
    """
    This function loads and returns a list of 4 images 
    witch name starts with the name base.
    
    Parameters
    ----------
    name_base : string
        The name base of the low resolution images to be loaded
    """
    
    L = []
    for i in range(4):
        filename = "{name_base}{index}.png".format(name_base = name_base, index = i)
        L.append(imageio.imread(filename)) # reading image return uint8 type array
        
    N = L[0].shape[0] # getting image low resolution side size

    return L,N

def enhancement(L,N,F,gamma):
    """
    This function apply enhancement method in the list of
    low resolution images.
    
    Parameters
    ----------
    L : list
        The list of size 4 containing the low quality images
    N : int
        The dimension of images in list
    F : int
        The enhancement function number that must be between 0 and 3
    gamma : int
        The parameter used in ehancement function number 3
    """
    
    # histogram functions
    
    no_levels = 256 # level of intensity (or color)
    
    def histogram(img):        
        hist = np.zeros(no_levels, dtype=int)
        
        for i in range(no_levels): 
            hist[i] = np.sum(img == i) # counting number of each intensity level
            
        return hist
    
    def individual_cumulative_histogram(img):
        hist = histogram(img) 
        # computing a cumulative histogram for each image
        for i in range(1,no_levels): 
            hist[i] += hist[i-1]

        return hist
    
    def cumulative_histogram_set():
        hist_c = np.zeros(no_levels, dtype=int)
        # computing a single cumulative histogram over all images in L
        for li in L: 
            hist = histogram(li)
            hist_c[0] += hist[0]
            for j in range(1,no_levels):
                hist_c[j] = hist[j] + hist_c[j-1]
                
        return hist_c
    
    #-----------------------------------------------------------------------
    
    def equalize(img,hist):
        img_eq = np.zeros((N,N), dtype=np.uint8)
        
        for z in range(no_levels): # computing equalized image's values
            s = ((no_levels-1)/float(N*N))*hist[z]
            img_eq[np.where(img == z)] = s
        
        return img_eq
    
    #-----------------------------------------------------------------------
    
    def gamma_correction_function(img):
        for x in range(N):
            for y in range(N):
                img[x,y] = math.floor(255*((img[x,y]/float(255))**(1/gamma)))
        
        return img
    
    #-----------------------------------------------------------------------
    
    L_enhan = L # F == 0 dont't need to apply any enhancement
    if F == 1:
        for i in range(4):
            hist = individual_cumulative_histogram(L[i])
            L_enhan[i] = equalize(L[i],hist)
    elif F == 2:
        hist = cumulative_histogram_set()
        for i in range(4):
            L_enhan[i] = equalize(L[i],hist) 
    elif F == 3:
        for i in range(4):
            L_enhan[i] = gamma_correction_function(L[i])
    
    return L_enhan

def superresolution(L,N):
    """
    This function combines the images in the list using 
    a composition method to return a high resolution image.
    
    Parameters
    ----------
    L : list
        The list of size 4 containing the low quality images
    N : int
        The dimension of images in list
    """
    
    M = N*2 # high resolution image have double of the low resolution
    H = np.zeros((M,M)) 
    x, y = 0, 0
    
    # iterating in each cell of all low resolution images to compose the high resolution image
    for i in range(N): 
        for j in range(N):
            H[x,y] = L[0][i,j]
            H[x,y+1] = L[1][i,j]
            H[x+1,y] = L[2][i,j]
            H[x+1,y+1] = L[3][i,j]
            y += 2
        x += 2
        y = 0
    
    return H,M

def compare_to_reference(img,ref_filename,M):
    """
    This function compare a image to a reference through
    Root Mean Squared Error (RMSE). 
    The higher the SRE, the greater the difference between
    the images.
    
    Parameters
    ----------
    img : array
        The source image array
    ref_filename : string
        The reference image filename
    M : int
        The dimension of high resoluton image
    """
    # loading image reference from file as float
    reference = imageio.imread(ref_filename).astype(float) 
    
    rmse = np.sum(np.square(reference-img))
    rmse = math.sqrt(rmse / (M*M))

    return round(rmse,4) # rounding to 4 decimal places

def enhancement_superresolution():
    """
    This function asks for some user inputs to read low resolution
    images from files, apply enhancement and superresolution methods 
    to increase resolution and, finally, compare to a high resolution
    image read from a file, printing the result.
    
    Inputs (in this order)
    ----------------------
    imglow : string
        The name base of the low resolution images
    imghigh : string
        The high resolution image filename
    F : int
        The enhancement function number that must be between 0 and 3
    gamma : int
        The parameter used in ehancement function number 3
    """
    
    # getting input parameters
    imglow = str(input().rstrip())
    imghigh = str(input().rstrip())
    F = int(input())
    gamma = float(input())
    
    L,N = load_images(imglow) # reading all low images
    L = enhancement(L,N,F,gamma) # applying the enhancement method in each image
    H,M = superresolution(L,N) # generating a high resolution image with the composition of the low image
    print(compare_to_reference(H,imghigh,M)) # print rmse result for comparison

if __name__ == "__main__":
    enhancement_superresolution()