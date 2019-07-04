import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

#Cambiamos el directorio de trabajo
os.chdir('/home/jose/Escritorio/VCP1/')

#Funcion para leer una imagen
#flagColor: 0 Grises, 1 Color
def leeimagen(filename,flagColor):
    #Cargamos la imagen
    img = cv2.imread(filename,flagColor)/255
    
    return img

# Funcion para mostrar una imagen.
def pintaI(im):
    #Mostramos la imagen
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    img = (np.clip(im,0,1)*255.).astype(np.uint8)
    cv2.imshow('image',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#Concatena varias imagenes
def multIM(img,nfil,ncol,tamx,tamy,color=True):
    fig=plt.figure(figsize=(tamx, tamy))
    for i,im in enumerate(img):
        fig.add_subplot(nfil, ncol, i+1)
        imgt = (np.clip(im,0,1)*255.).astype(np.uint8)
        if color:
            nimg = cv2.cvtColor(imgt, cv2.COLOR_BGR2RGB)
        else:
            nimg = cv2.cvtColor(imgt,cv2.COLOR_GRAY2RGB)
        plt.imshow(nimg)
    plt.show()
    
#Funcion que normaliza los valores de una imagen a valores positivos
def transformar(img):
    minimo = np.min(img)
    maximo = np.max(img)
    
    if minimo < 0 or maximo > 1:
        img = (img-minimo)/(maximo-minimo)
    
    return img

#Funcion para el ejercicio 2.A que nos devuelve una imagen con una convolucion
# separable aplicada indicando el tamaño del kernel
# =============================================================================
# tam debe ser impar
# =============================================================================
def convolucion_gauss(img,tam):
    val = cv2.getGaussianKernel(ksize=tam,sigma=-1)
    
    res = cv2.sepFilter2D(src=img,ddepth=-1,kernelX=val,kernelY=val,borderType=cv2.BORDER_REFLECT_101)
    
    return res

#Funcion para el ejercicio 2.B que nos devuelve una imagen con una convolucion
# 2D de primera derivada aplicada indicando el tamaño del kernel
# =============================================================================
# tam debe ser impar
# =============================================================================
def convolucion_1deriv(img,tam):
    kx,ky = cv2.getDerivKernels(1,1,tam)
    
    res = cv2.sepFilter2D(src=img,ddepth=-1,kernelX=kx,kernelY=ky,borderType=cv2.BORDER_CONSTANT)
    
    return res

#Funcion para el ejercicio 2.C que nos devuelve una imagen con una convolucion
# 2D de segunda derivada aplicada indicando el tamaño del kernel
# =============================================================================
# tam debe ser impar
# =============================================================================
def convolucion_2deriv(img,tam):
    kx,ky = cv2.getDerivKernels(2,2,tam)
    
    res = cv2.sepFilter2D(src=img,ddepth=-1,kernelX=kx,kernelY=ky)
    
    return res

# Funcion para el ejercicio 2.D que nos devuelve una imagen compuesta por
# 4 niveles de la piramide Gaussiana
def piramide_gauss(img):
    res = [img]
    
    aux = cv2.pyrDown(img,borderType=cv2.BORDER_REFLECT_101)
    res.append(aux)
    aux = cv2.pyrDown(aux,borderType=cv2.BORDER_REFLECT_101)
    res.append(aux)
    aux = cv2.pyrDown(aux,borderType=cv2.BORDER_REFLECT_101)
    res.append(aux)
    
    return res


#Funcion que muestra los elementos de una piramide
def display_piramide(img,color=True):
    
    for im in img:
        imgt = (np.clip(im,0,1)*255.).astype(np.uint8)
        if color:
            nimg = cv2.cvtColor(imgt, cv2.COLOR_BGR2RGB)
        else:
            nimg = cv2.cvtColor(imgt,cv2.COLOR_GRAY2RGB)
        dpi = 50
        height, width, depth = nimg.shape
    
        # What size does the figure need to be in inches to fit the image?
        figsize = width / float(dpi), height / float(dpi)
    
        # Create a figure of the right size with one axes that takes up the full figure
        plt.figure(figsize=figsize)
        #ax = fig.add_axes([0, 0, 1, 1])
    
        # Hide spines, ticks, etc.
        #ax.axis('off')
    
        # Display the image.
        plt.imshow(nimg)
    plt.show()

#Funcion del ejercicico 2.D que devuelve tanto las imagenes que se almacenan
# para la piramide gaussiana como las imagenes en sí
def piramide_laplaciana_down(img):
    res = [img]
    dif = []
    aux = img
    
    for i in range(3):
        mini = cv2.pyrDown(aux,borderType=cv2.BORDER_REFLECT_101)
        col,fil = aux.shape
        maxi = cv2.pyrUp(mini,None,(fil,col),borderType=cv2.BORDER_REFLECT_101)
        dif.append(aux-maxi)
        res.append(mini)
        aux = mini
    
    return res,dif

def piramide_laplaciana_up(img,dif):
    res = [img]
    aux = img
    
    for im in reversed(dif):
        col,fil = im.shape
        maxi = cv2.pyrUp(aux,None,(fil,col),borderType=cv2.BORDER_REFLECT_101)
        real = maxi+im
        res.append(real)
        aux = real
    
    return res


#Funcion que crea una imagen hibrida
def crear_hibrida(img1,img2,ksize1,ksize2):
    gauss = cv2.getGaussianKernel(ksize1,-1)
    gauss = cv2.sepFilter2D(img1,-1,gauss,gauss)
    
    g2 = cv2.getGaussianKernel(ksize2,-1)
    g2 = cv2.sepFilter2D(img2,-1,g2,g2)
    laplacian = img2-g2
    
    res = gauss+laplacian
    
    return res

###############################################################################
###############################################################################
###############################################################################
###############################################################################


# =============================================================================
# Ejercicio 1: A)
# El cálculo de la convolución de una imagen con una máscara
# Gaussiana 2D (Usar GaussianBlur). Mostrar ejemplos con distintos
# tamaños de máscara y valores de sigma. Valorar los resultados.
# =============================================================================
    
img = leeimagen('imagenes/bicycle.bmp',1)

concat = [img]

for i in range(3,12,2):
    concat.append(cv2.GaussianBlur(img,(i,i),0,0))

multIM(concat,2,3,28,13)

input()

# =============================================================================
# Ejercicio 1: B)
# Usar getDerivKernels para obtener las máscaras 1D que permiten
# calcular al convolución 2D con máscaras de derivadas. Representar
# e interpretar dichas máscaras 1D para distintos valores de sigma.
# =============================================================================

for i in range(3,8,2):
    kx, ky = cv2.getDerivKernels(1,0,i)

    print("Kernel de tamaño "+str(i)+ " con dx=1 y dy=0")
    print(np.dot(kx,np.transpose(ky)))

for i in range(3,8,2):
    kx, ky = cv2.getDerivKernels(1,1,i)

    print("Kernel de tamaño "+str(i)+ " con dx=1 y dy=1")
    print(np.dot(kx,np.transpose(ky)))
    
for i in range(3,8,2):
    kx, ky = cv2.getDerivKernels(0,1,i)

    print("Kernel de tamaño "+str(i)+ " con dx=0 y dy=1")
    print(np.dot(kx,np.transpose(ky)))
    
input()


# =============================================================================
# Ejercicio 1: C)
# Usar la función Laplacian para el cálculo de la convolución 2D con
# una máscara de Laplaciana-de-Gaussiana de tamaño variable.
# Mostrar ejemplos de funcionamiento usando dos tipos de bordes y
# dos valores de sigma: 1 y 3.
# =============================================================================

dst = []

aux	= cv2.Laplacian(src=img,ddepth=-1,ksize=3,borderType=cv2.BORDER_REFLECT_101)
dst.append(aux)
aux	= cv2.Laplacian(src=img,ddepth=-1,ksize=3,borderType=cv2.BORDER_CONSTANT)
dst.append(aux)

aux	= cv2.Laplacian(src=img,ddepth=-1,ksize=17,borderType=cv2.BORDER_REFLECT_101)
dst.append(aux)
aux	= cv2.Laplacian(src=img,ddepth=-1,ksize=17,borderType=cv2.BORDER_CONSTANT)
dst.append(aux)

dst = map(transformar,dst)

multIM(dst,2,2,15,10)

input()

# =============================================================================
# Ejercicio 2: A)
# El cálculo de la convolución 2D con una máscara separable de
# tamaño variable. Usar bordes reflejados. Mostrar resultados
# =============================================================================

img = leeimagen('imagenes/motorcycle.bmp',0)

conv_gauss1 = convolucion_gauss(img,7)
conv_gauss2 = convolucion_gauss(img,15)
conv_gauss3 = convolucion_gauss(img,21)

multIM([conv_gauss1,conv_gauss2,conv_gauss3],1,3,20,10,False)

input()

# =============================================================================
# Ejercicio 2: B)
# El cálculo de la convolución 2D con una máscara 2D de 1a 1a
# derivada de tamaño variable. Mostrar ejemplos de
# funcionamiento usando bordes a cero.
# =============================================================================

conv_1deriv1 = transformar(convolucion_1deriv(img,7))
conv_1deriv2 = transformar(convolucion_1deriv(img,15))
conv_1deriv3 = transformar(convolucion_1deriv(img,21))

multIM([conv_1deriv1,conv_1deriv2,conv_1deriv3],1,3,20,10,False)

input()

# =============================================================================
# Ejercicio 2: C)
# El cálculo de la convolución 2D con una máscara 2D de 2a
# derivada de tamaño variable.
# =============================================================================

conv_2deriv1 = transformar(convolucion_2deriv(img,7))
conv_2deriv2 = transformar(convolucion_2deriv(img,15))
conv_2deriv3 = transformar(convolucion_2deriv(img,21))

multIM([conv_2deriv1,conv_2deriv2,conv_2deriv3],1,3,20,10,False)

input()

# =============================================================================
# Ejercicio 2: D)
# Una función que genere una representación en pirámide
# Gaussiana de 4 niveles de una imagen. Mostrar ejemplos de
# funcionamiento usando bordes
# =============================================================================

dst = piramide_gauss(img)

display_piramide(dst,False)

input()

# =============================================================================
# Ejercicio 2: E)
# Una función que genere una representación en pirámide
# Laplaciana de 4 niveles de una imagen. Mostrar ejemplos de
# funcionamiento usando bordes.
# =============================================================================

dest = []
dif = []

dst,dif = piramide_laplaciana_down(img)

display_piramide(dst,False)

trans = map(transformar,dif)

display_piramide(trans,False)

up = piramide_laplaciana_up(dst[len(dst)-1],dif)

display_piramide(up,False)

input()

# =============================================================================
# Ejercicio 3
# Mezclando adecuadamente una parte de las frecuencias altas de una
# imagen con una parte de las frecuencias bajas de otra imagen, obtenemos
# una imagen híbrida que admite distintas interpretaciones a distintas
# distancias ( ver hybrid images project page).
# Para seleccionar la parte de frecuencias altas y bajas que nos quedamos
# de cada una de las imágenes usaremos el parámetro sigma del
# núcleo/máscara de alisamiento gaussiano que usaremos. A mayor valor
# de sigma mayor eliminación de altas frecuencias en la imagen
# convolucionada. Para una buena implementación elegir dicho valor de
# forma separada para cada una de las dos imágenes ( ver lase comendaciones 
# dadas en el paper de Oliva et al.). Recordar que las máscaras 1D siempre deben
# tener de longitud un número impar.
# 
# Implementar una función que genere las imágenes de baja y alta
# frecuencia a partir de las parejas de imágenes ( solo en la versión de
# imágenes de gris) . El valor de sigma más adecuado para cada pareja
# habrá que encontrarlo por experimentación
# 
# Escribir una función que muestre las tres imágenes ( alta,
# baja e híbrida) en una misma ventana. (Recordar que las
# imágenes después de una convolución contienen número
# flotantes que pueden ser positivos y negativos)
# 
# Realizar la composición con al menos 3 de las parejas de
# imágenes
# =============================================================================

img1 = leeimagen('imagenes/cat.bmp',0)
img2 = leeimagen('imagenes/dog.bmp',0)

animales = crear_hibrida(img1,img2,41,31)

multIM([img1,img2,animales],1,3,15,10,False)

img1 = leeimagen('imagenes/einstein.bmp',0)
img2 = leeimagen('imagenes/marilyn.bmp',0)

personas = crear_hibrida(img1,img2,21,17)

multIM([img1,img2,personas],1,3,15,10,False)

img1 = leeimagen('imagenes/bird.bmp',0)
img2 = leeimagen('imagenes/plane.bmp',0)

aire = crear_hibrida(img1,img2,24,17)

multIM([img1,img2,aire],1,3,15,10,False)

pir = piramide_gauss(personas)

display_piramide(pir,False)

input()
