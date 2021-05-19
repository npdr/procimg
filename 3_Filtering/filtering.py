# Trabalho 3 - Filtragem
# SCC0251 - Image Processing (01/2021)
# Fabiana Dalacqua Mendes - 9894536
# Pedro Henrique Nieuwenhoff - 10377729
import numpy as np
import imageio
import math

def load_image(filename):
    """
    This function loads and returns a image.
    
    Parameters
    ----------
    filename : string
        The image filename
    """
    
    img = imageio.imread(filename) # reading image return uint8 type array
    
    return img

def filtering(img,F,n):
    """
    This function return a filtered image using
    a filtering method.
    
    Parameters
    ----------------------
    img : array
        The source image array
    F : int
        The method identifier between 1 and 3
    n : int
        The lateral size of filter
    """
    
    def filtering_1D(w):
        flat_img = img.flatten() # formating 2D array to 1D
        l = len(flat_img) # getting flat image lenght
        
        img_filt = np.empty(l) # initializing result image
        
        # padding vector using wrap technique
        pad = int((n-1)/2)
        pad_img = np.pad(flat_img,(pad,pad),mode='wrap') 
        
        # applying filter for each cell image
        x = 0 
        for i in range(pad,l):
            sub_image = pad_img[i-pad:i+pad+1]
            img_filt[x] = np.sum(np.multiply(sub_image,w))
            x += 1
        
        N,M = img.shape
        
        return np.reshape(img_filt,(N,M)) # reshaping 1D array to 2D
        
     
    #-----------------------------------------
    
    def filtering_2D(w):
        N,M = img.shape
        
        img_filt = np.empty((N,M)) # initializing result image
        
        # padding vector using reflect technique
        pad = int((n-1)/2)
        pad_img = np.pad(img,((pad,pad),(pad,pad)),mode='reflect') 
        
        # applying filter for each cell image
        x, y = 0, 0 
        for i in range(pad,N):
            for j in range(pad,M):
                sub_image = pad_img[i-pad:i+pad+1,j-pad:j+pad+1]
                img_filt[x,y] = np.sum(np.multiply(sub_image,w))
                y += 1
            x += 1
            y = 0
          
        return img_filt
     
    #-----------------------------------------
    
    def get_median(sub_image,med_index):
        sort_img = np.sort(sub_image,axis=None)
        return sort_img[med_index]
    
    def median_filter_2D():
        N,M = img.shape
        
        img_filt = np.empty((N,M)) # initializing result image
        
        # padding vector with zeros
        pad = int((n-1)/2)
        pad_img = np.pad(img,((pad,pad),(pad,pad)))
        
        med_index = math.floor((n*n)/2)
        
        # applying filter for each cell image
        x, y = 0, 0 
        for i in range(pad,N):
            for j in range(pad,M):
                sub_image = pad_img[i-pad:i+pad+1,j-pad:j+pad+1]
                img_filt[x,y] = get_median(sub_image,med_index)
                y += 1
            x += 1
            y = 0
          
        return img_filt
    
    #-----------------------------------------
    
    img_filt = img
    
    if F == 1:
        w = np.array([float(xi) for xi in input().split()]) # getting the sequence of n weights

        img_filt = filtering_1D(w) # applying filtering
    elif F == 2:
        aux = [] # getting the n rows of n filter weights
        for i in range(n):
            aux.append(np.array([float(xi) for xi in input().split()]))
    
        w = np.array(aux)
        img_filt = filtering_2D(w) # applying filtering
    elif F == 3:
        img_filt = median_filter_2D() # applying filtering
    
    return img_filt

def normalize(img):
    """ 
    This function return a normalized image with values between zero and max value.
    
    Parameters
    ----------
    img : array
        The source image array
    max_norm : int
        The max value usually based on power of 2
    """
    
    imin = np.min(img)
    imax = np.max(img)
    
    img_norm = (img - imin)/(imax - imin) # normalizing between 0 and 1 
    
    return (img_norm * 255).astype(np.uint8) # normalizing to 0 - 255

def rmse_compare(img_a, img_b):
    """
    This function compare images through Root Mean Squared Error (RMSE). 
    The higher the SRE, the greater the difference betweenthe images.
    
    Parameters
    ----------
    img_a : array
        The first image array
    img_b : array
        The second image array
    """
    
    # getting images resolution 
    N,M = img_a.shape # both images have the same size
    
    # converting the matrices to int32, allowing negative difference
    img_a = img_a.astype(np.int32)
    img_b = img_b.astype(np.int32)
    
    # computing rsme
    rmse = np.sum(np.square(img_a-img_b))
    rmse = math.sqrt(rmse / (N*M))

    return rmse

def image_filtering():
    """
    This function asks for some user inputs to read image
    from file, apply filtering method and compare to the
    original image, printing the result.
    
    Inputs (in this order)
    ----------------------
    imgref : string
        The image filename
    F : int
        The method identifier between 1 and 3
    n : int
        The lateral size of filter
    """
    
    # getting inputs
    imgref = str(input().rstrip())
    F = int(input())
    n = int(input())

    img = load_image(imgref) # reading image

    img_filt = filtering(img,F,n) # filtering image

    img_filt = normalize(img_filt) # normalizing filtered image
    
    print(rmse_compare(img,img_filt)) # print rmse result for comparison

if __name__ == "__main__":
    image_filtering()