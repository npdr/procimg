# Trabalho 1 - Geração de imagens
# SCC0251 - Image Processing (01/2021)
# Fabiana Dalacqua Mendes - 9894536
# Pedro Henrique Nieuwenhoff - 10377729
import numpy as np
import random
import math

def normalize(img,max_norm):
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
    return img_norm * (max_norm - 1) # normalizing to new max based on power of 2

def generate_image(C,F,Q,S):
    """
    This function return a image synthesized based on math or random function.
    
    Parameters
    ----------
    C : int
        The side size of the square image to be sintetized
    F : int
        The function number that must be between 1 and 5
    Q : int
        The parameter used for image generation
    S : int
        The seed for functions based on random values
    """
    # defining 1-5 functions for image synthetise
    def function_1(x,y):
        return (x*y + 2*y)

    def function_2(x,y,Q):
        return abs(np.cos(x/Q) + 2 * np.sin(y/Q))

    def function_3(x,y,Q):
        return abs(3 * (x/Q) - np.cbrt(y/Q))

    def function_4(C,S):
        img = np.zeros((C,C), dtype=float) # initializing array with zeros
        random.seed(S) # setting seed for random function
        
        for y in range(C):
            for x in range(C):
                img[x,y] = random.random()
                
        return img

    def function_5(C,S):
        img = np.zeros((C,C), dtype=float) # initializing array with zeros
        random.seed(S) # setting seed for random function
        
        x, y = 0, 0
        steps = 1 + C*C

        for i in range(steps): # randomwalking the image grid
            img[x,y] = 1
            dx = random.randint(-1,1)
            dy = random.randint(-1,1)
            x = (x + dx) % C
            y = (y + dy) % C
            
        return img
    
    # synthetizing image according to the choose function
    if F == 1:
        img_sint = np.fromfunction(lambda x,y: function_1(x,y), (C,C))
    elif F == 2:
        img_sint = np.fromfunction(lambda x,y: function_2(x,y,Q), (C,C))
    elif F == 3:
        img_sint = np.fromfunction(lambda x,y: function_3(x,y,Q), (C,C))
    elif F == 4:
        img_sint = function_4(C,S)
    elif F == 5:
        img_sint = function_5(C,S)
    else:
        img_sint = None
    
    # normalizing image to [0, 2ˆ16 - 1]
    img_sint_norm = normalize(img_sint, 65536)
    
    return img_sint_norm

def digitalize(img,C,N,B):
    """
    This function simulates the digitalization of an image, using a 
    downsampling method follow by a quantization through bitwise shift.
    Parameters
    ----------
    img : array
        The source image array
    C : int
        The side size of the square image to be sintetized
    N : int
        The side size of the sampled image
    B : int
        The bits number to bitwise shift
    """
    def downsampling(img,C,N):
        img_samp = np.zeros((N,N),float)
        
        step = math.floor(C/N)
        x, y = 0, 0
        for x_samp in range(N):
            for y_samp in range(N):
                img_samp[x_samp,y_samp] = img[x,y]
                y += step
            
            x += step
            y = 0
        
        return img_samp

    def quantization(img,B):
        img_norm = normalize(img,256).astype(np.uint8) # normalizing to [0, 2ˆ8 - 1]
        
        b = 8 - B # bitwise shift
        img_quant = (img_norm >> b) << b # shift right follow by shift left
        
        return img_quant
    
    # digitalizing image 
    img_samp = downsampling(img,C,N)
    img_dig = quantization(img_samp,B)

    return img_dig

def compare_to_reference(img,r):
    """
    This function compare a ganerated image to a reference through
    Square Root Error (SRE). 
    The higher the SRE, the greater the difference between the images.
    SRE equals zero means the images are the same.
    Parameters
    ----------
    img : array
        The source image array
    r : string
        The reference image filename
    """
    reference = np.load(r) # loading image reference from file
    
    sre = np.sum(np.square(img-reference))
    sre = math.sqrt(sre) # calculating square root error

    return round(sre,4) # rounding to 4 decimal places

def image_generator():
    """
    This function asks for some user inputs to generate image 
    and compare to a reference image read from a file,
    printing the result.
    
    Inputs (in this order)
    ----------------------
    r : string
        The reference image filename
    C : int
        The side size of the square image to be sintetized
    F : int
        The function number that must be between 1 and 5
    Q : int
        The parameter used for image generation
    N : int
        The side size of the sampled image
    B : int
        The bits number to bitwise shift
    S : int
        The seed for functions based on random values
    """
    # getting input parameters
    r = str(input().rstrip())
    C = int(input())
    F = int(input())
    Q = int(input())
    N = int(input())
    B = int(input())
    S = int(input())

    image = generate_image(C,F,Q,S) # generating image
    image = digitalize(image,C,N,B) # digitalizing
    print(compare_to_reference(image, r)) # print rse value for comparison

if __name__ == "__main__":
    image_generator()