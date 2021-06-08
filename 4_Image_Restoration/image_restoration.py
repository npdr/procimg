# Trabalho 4 - Image Restoration
# SCC0251 - Image Processing (01/2021)
# Fabiana Dalacqua Mendes - 9894536
# Pedro Henrique Nieuwenhoff - 10377729
import numpy as np
import imageio
import math
from scipy.fftpack import fftn, ifftn, fftshift

def restore(img,F,gamma):
    """
    This function apply the restoration filter selected
    and return an restored image.
    
    Parameters
    ----------
    img : uint8 array
        The degraded image array
    F : int
        TThe type of filter to perform image restoration (1 (denoising) or 2 (constrained l.s.))
    gamma : int
        The parameter used by either filter
    """
    
    
    # returns the arithmetic mean and the standard deviation
    def average_metrics(image):
        return np.mean(image), np.std(image)
    
    # returns the median and interquartile range
    def robust_metrics(image):
        img_sorted = np.sort(image, axis=None)
        n = (img_sorted.shape[0] - 1) // 2
        # calculates the interquartile range
        Q1 = img_sorted[(n + 1) // 2]
        Q3 = img_sorted[n + (n + 1) // 2]

        centr = float(Q3 - Q1)
        median = float(img_sorted[n])

        return median, centr
    
    """
    This function restore an image using Adaptive Denoising Filtering
    
    Parameters
    ----------
    coord : int array
        The coordinates to crop image and compute the estimated noise dispersion
    n : int
        The size of the filter
    denoise : string
        The denoising mode 
    """
    def adaptive_denoising(coord,n,denoise):
        N, M = img.shape
        res_img = np.zeros(img.shape, dtype=float)
        
        pad = (n-1)//2
        pad_img = np.pad(img,((pad,pad),(pad,pad)),mode='symmetric') 
        
        # cropping image with coordinates
        img_crop = pad_img[coord[0]:coord[1],coord[2]:coord[3]]

        # setting the configuration needed for each denoising mode
        if denoise == 'average':
            get_metrics = average_metrics
            disp_e = np.std(img_crop)
        elif denoise == 'robust':
            get_metrics = robust_metrics
            Q3, Q1 = np.percentile(img_crop,[75, 25])
            disp_e = float(Q3 - Q1)
        
        # avoid having the filter doing nothing for the whole image
        if disp_e == 0: disp_e = 1
        
        # applying filter
        x, y = 0, 0 
        for i in range(pad,N+pad):
            for j in range(pad,M+pad):
                sub_img = pad_img[i-pad:i+pad+1,j-pad:j+pad+1] # getting sub image
                
                centr_l, disp_l = get_metrics(sub_img) # getting the denoising metrics
                
                if disp_l == 0: # avoiding division by zero
                    disp_l = disp_e 
                    centr_l = 0          
                
                # computing the restored pixel with adaptive filter
                res_img[x,y] = pad_img[i,j] - gamma * (disp_e/disp_l) * (pad_img[i,j] - centr_l)
                y += 1
            x += 1
            y = 0

        np.clip(res_img,0,255) # clipping image array
        
        return res_img
        
    #---------------------------------------
    
    # gaussian filter provided in project description file
    def gaussian_filter(k=3,sigma=1.0):
        arx = np.arange((-k // 2) + 1.0, (k // 2) + 1.0) 
        x, y = np.meshgrid(arx, arx)
        filt = np.exp(-(1/2)*(np.square(x) + np.square(y))/np.square(sigma))
        
        return filt/np.sum(filt)
    
    def get_gaussian_filter(k,sigma):
        # getting the filter that caused the degradation
        gaussian = gaussian_filter(k, sigma)
        
        # padding the gaussian filter
        n = img.shape[0] // 2 - gaussian.shape[0] // 2
        pad_gaussian = np.pad(gaussian,(n,n-1),'constant')
        
        return pad_gaussian
    
    # return the lapaclian filter provides in project description file
    def get_inverse_laplacian_filter():
        p = np.array([[0,-1,0],[-1,4,-1],[0,-1,0]])
        n = img.shape[0] // 2 - p.shape[0] // 2
        P = np.pad(p, (n, n-1), 'constant')
        
        return P
        
    """
    This function restore an image using Constrained Least Squares Filtering
    
    Parameters
    ----------
    k : int, sigma : float
        The values used to compute the gaussian filter
    """
    def constrained_least_squares(k,sigma):
        # getting the fourier transform of image
        G = fftn(img)
        
        # getting the gaussian filter
        H = fftn(get_gaussian_filter(k,sigma))
        cH = np.conj(H) # computes the complex conjugate of H

        # getting the fourier transform of laplacian filter
        P = fftn(get_inverse_laplacian_filter())

        # computing the constrained least squares
        F = (cH/(np.square(np.abs(H)) + gamma * np.square(np.abs(P))))
        R = np.multiply(F,G)

        # computing the fourier transform
        res_img = np.real(fftshift(ifftn(R)))
        
        np.clip(res_img,0,255) # clipping image array
        
        return res_img
    
    #---------------------------------------
    
    if F == 1: 
         # getting coordinates of a flat rectangle in the image
        coord = np.array([int(xi) for xi in input().split()])
        n = int(input()) # getting the size of the filter
        denoise = str(input().rstrip()) # getting the denoise mode (average or robust)
        
        return adaptive_denoising(coord,n,denoise)
    elif F == 2:
        n = int(input()) # getting the size of the filter
        sigma = float(input())
        
        return constrained_least_squares(n,sigma)

def rmse_compare(f,g):
    """
    This function compare images through Root Mean Squared Error (RMSE). 
    The higher the SRE, the greater the difference between the images.
    
    Parameters
    ----------
    img_a : array
        The first image array
    img_b : array
        The second image array
    """
    
    # getting images resolution 
    N,M = f.shape # both images have the same size
    
    # converting the matrices to int32, allowing negative difference
    f = f.astype(np.int32)
    g = g.astype(np.int32)
    
    # computing rsme
    rmse = np.sum(np.square(f-g))
    rmse = math.sqrt(rmse / (N*M))

    return rmse

def image_restoration():
    """
    This function asks for some user inputs to read 
    degraded image from file, apply an restoration method
    and compare to reference image, printing the result.
    
    Inputs (in this order)
    ----------------------
    ref_img : string
        The filename for the reference image
    deg_img : string
        The filename for the degraded image
    F : int
        The type of filter to perform image restoration (1 (denoising) or 2 (constrained l.s.))
    gamma : int
        The parameter used by either filter
    """
    
    # getting input parameters
    ref = str(input().rstrip())
    deg = str(input().rstrip())
    F = int(input())
    gamma = float(input())
    
    # reading images as uint8 array
    ref_img = imageio.imread(ref)
    deg_img = imageio.imread(deg)
    
    res_img = restore(deg_img,F,gamma) # applying restoration method in degraded image
    print(rmse_compare(ref_img, res_img)) # comparing through rmse value

if __name__ == "__main__":
    image_restoration()