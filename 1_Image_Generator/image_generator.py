#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Trabalho 1 - Geração de imagens
# SCC0251 - Image Processing (01/2021)
# Fabiana Dalacqua Mendes - 9894536
# Pedro Henrique Nieuwenhoff - 10377729
import numpy as np
import numpy.random as random
np.set_printoptions(suppress=True)
import math


# In[ ]:


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
    img_norm = (img - imin)/(imax - imin) # normalizes between 0 and 1
    return img_norm * (max_norm - 1)


# In[ ]:


def generate_image(C,F,Q,S):
    """
    This function return a image synthesize based on math or random function.
    
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
    # initializing array with zeros
    img_sint = np.zeros((C,C), float)
    
    # 1-5 functions definitions
    def function_1(x,y):
        return (x*y + 2*y)

    def function_2(x,y,Q):
        return abs(np.cos(x/Q) + 2 * np.sin(y/Q))

    def function_3(x,y,Q):
        return abs(3 * (x/Q) - np.cbrt(y/Q))

    def function_4():
        return random.random()

    def function_5(x,dx,C):
        return (x + dx) % C
    
    # synthetizing image according to the choose function
    if F == 1:
        img_sint = np.fromfunction(lambda x,y: function_1(x,y), (C,C))
    elif F == 2:
        img_sint = np.fromfunction(lambda x,y: function_2(x,y,Q), (C,C))
    elif F == 3:
        img_sint = np.fromfunction(lambda x,y: function_3(x,y,Q), (C,C))
    elif F == 4:
        random.seed(S)
        for y in range(C):
            for x in range(C):
                img_sint[x,y] = function_4()
    elif F == 5:
        random.seed(S)
        x, y = 0, 0
        img_sint[x,y] = 1
        steps = 1 + C**2
    
        for i in range(steps):
            dx = random.randint(-1,1)
            dy = random.randint(-1,1)
            x = function_5(x,dx,C)
            y = function_5(y,dy,C)
            img_sint[x,y] = 1
            
        plt.imshow(img_sint, cmap="gray")
    else:
        img_sint = None
    
    # normalizing image to 2ˆ16
    img_sint_norm = normalize(img_sint, 65536)
    
    return img_sint_norm


# In[ ]:


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
        img_norm = normalize(img,256).astype(np.uint8)
        
        b = 8 - B
        img_quant = (img_norm >> b) << b
        return img_quant
    
    # sampling image and digitalizing
    img_samp = downsampling(img,C,N)
    img_dig = quantization(img_samp,B)

    return img_dig


# In[ ]:


def compare_to_reference(img,r):
    """
    This function compare a ganerated image to a reference through
    Square Root Error (SRE). 
    The higher the SRE, the greater the difference between the images.
    Parameters
    ----------
    img : array
        The source image array
    r : string
        The reference image filename
    """
    reference = np.load(r) # carrega a imagem de referência
    
    sre = np.sum(np.square(img-reference))
    sre = math.sqrt(sre)
    
    return round(sre,4)


# In[ ]:


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
    r = str(input().rstrip())
    C = int(input())
    F = int(input())
    Q = int(input())
    N = int(input())
    B = int(input())
    S = int(input())

    image = generate_image(C,F,Q,S)
    image = digitalize(image,C,N,B)
    print(compare_to_reference(image, r))


# In[ ]:


if __name__ == "__main__":
    image_generator()

