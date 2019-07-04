#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 19 16:57:26 2018

@author: jose
"""

import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pickle
from os import listdir
from os.path import join,isfile
from numpy.linalg import norm

#Cambiamos el directorio de trabajo
os.chdir('./imagenes/')

def loadDictionary(filename):
    with open(filename,"rb") as fd:
        feat=pickle.load(fd)
    return feat["accuracy"],feat["labels"], feat["dictionary"]

def loadAux(filename, flagPatches):
    if flagPatches:
        with open(filename,"rb") as fd:
            feat=pickle.load(fd)
        return feat["descriptors"],feat["patches"]
    else:
        with open(filename,"rb") as fd:
            feat=pickle.load(fd)
        return feat["descriptors"]

def extractRegion(image):
    global refPt, imagen,FlagEND
    imagen=image.copy()
    # load the image and setup the mouse callback function
    refPt=[]
    FlagEND=True
    #image = cv2.imread(filename)
    cv2.namedWindow("image")
    # keep looping until the 'q' key is pressed
    cv2.setMouseCallback("image", click_and_draw)
    #
    while FlagEND:
    	# display the image and wait for a keypress
        cv2.imshow("image", image)
        cv2.waitKey(0)
    #
    print('FlagEND', FlagEND)
    refPt.pop()
    refPt.append(refPt[0])
    cv2.destroyWindow("image")
    return refPt	

def click_and_draw(event,x,y,flags,param):
    global refPt, imagen,FlagEND
    
    
   # if the left mouse button was clicked, record the starting
	# (x, y) coordinates and indicate that cropping is being
	# performed
    if  event == cv2.EVENT_LBUTTONDBLCLK:
        FlagEND= False
        cv2.destroyWindow("image")
        
    elif event == cv2.EVENT_LBUTTONDOWN:
        refPt.append((x, y))
        #cropping = True
        print("rfePt[0]",refPt[0])
    

    elif (event == cv2.EVENT_MOUSEMOVE) & (len(refPt) > 0) & FlagEND:
    # check to see if the mouse move
        clone=imagen.copy()
        nPt=(x,y)
        print("npt",nPt)
        sz=len(refPt)
        cv2.line(clone,refPt[sz-1],nPt,(0, 255, 0), 2)
        cv2.imshow("image", clone)
        cv2.waitKey(0)
        
    elif event == cv2.EVENT_RBUTTONDOWN:
		# record the ending (x, y) coordinates and indicate that
		# the cropping operation is finished
        refPt.append((x, y))
        #cropping = False
        sz=len(refPt)
        print("refPt[sz]",sz,refPt[sz-1])
        cv2.line(imagen,refPt[sz-2],refPt[sz-1],(0, 255, 0), 2)
        cv2.imshow("image", imagen)
        cv2.waitKey(0)

# =============================================================================
# Funcion para leer una imagen
# flagColor: 0 Grises, 1 Color
# =============================================================================
def leeimagen(filename,flagColor):
    #Cargamos la imagen
    img = cv2.imread(filename,flagColor)
    
    return img

# =============================================================================
# Concatena varias imagenes
# =============================================================================
def multIM(img,nfil,ncol,tamx,tamy,color=True):
    fig=plt.figure(figsize=(tamx, tamy))
    for i,im in enumerate(img):
        fig.add_subplot(nfil, ncol, i+1)
#        imgt = (np.clip(im,0,1)*255.).astype(np.uint8)
        if color:
            nimg = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        else:
            nimg = cv2.cvtColor(im,cv2.COLOR_GRAY2RGB)
        plt.imshow(nimg)
    plt.show()

# =============================================================================
# Funcion que crea una mascara dada una imagen y los vertices de la zona
# donde queremos crea la mascara
# =============================================================================
def crearMascara(img,vertices):
    #Creamos una matriz vacia
    m = np.zeros(img.shape,dtype=np.uint8)
    
    #Creamos un array con los puntos seleccionados de la imagen
    approCurve = cv2.approxPolyDP(np.array(vertices,dtype=np.int64),1.0,True)
    
    #Ahora pasamos estos puntos dentro de la matriz creada al inicio.
    cv2.fillConvexPoly(m, approCurve,color=(255,255,255))
    
    #Pasamos el color de la mascara a grises
    m = cv2.cvtColor(m,cv2.COLOR_BGR2GRAY)
    
    #Devolvemos la mascara
    return m
    
# =============================================================================
# Calcula los descriptores de una imagen
# =============================================================================
def calcularDescriptores(img,masc=None):
    #Pasamos la imagen a gris
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
    #Calculamos los descriptores usando SIFT
    sift = cv2.xfeatures2d.SIFT_create()
    kp,desc = sift.detectAndCompute(gray,masc)

    #Devolvemos tanto los keypoints como los descriptores
    return [kp,desc]
    
# =============================================================================
# Calcula las correspondencias entre dos descriptores usando el algoritmo 2NN-Low
# =============================================================================
def BFL2NN(des1,des2):
    bf = cv2.BFMatcher(normType=cv2.NORM_L2,crossCheck=False)

    matches = bf.knnMatch(des1,des2,k=2)
    
    good = []
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            good.append([m])
        
    good = sorted(good,key=lambda dist:dist[0].distance, reverse=False)
    
    return good

# =============================================================================
# Calcula las correspondencias entre dos imagenes usando una mascara
# =============================================================================
def calcularCorrespondencias(img1,img2,masc):
    # Calculamos los descriptores.
    kp1, desc1 = calcularDescriptores(img1, masc)
    kp2,desc2 = calcularDescriptores(img2)
    
    # Calculamos las correpondencias.
    corr = BFL2NN(desc1,desc2)

    # Dibujamos las correspondencias.
    matches = cv2.drawMatchesKnn(img1=img1,keypoints1=kp1,img2=img2,keypoints2=kp2,matches1to2=corr,outImg=None,flags=2)
    
    return matches
    
# =============================================================================
# Funcion que calcula los votos para todas las palabras de un diccionario dada
# una imagen
# =============================================================================
def calcularVotos(img,diccionario):
    #Calculamos os decriptores de la imagen.
    kp, desc = calcularDescriptores(img)

    # Normalizamos los descriptores de la imagen.
    desc = cv2.normalize(src=desc,dst=None,norm_type=cv2.NORM_L2)
    dicc = cv2.normalize(src=diccionario,dst=None,norm_type=cv2.NORM_L2)

    # calculamos de semejanza entre las palabras y los descriptores.
    votos = np.dot(dicc,desc.T)


    # Calculamos las palabras que han sido más votadas para la imagen.
    palabrasImg = np.zeros(shape=votos.shape[0], dtype=np.int)
    for col in range(votos.shape[1]):
        # Obtenemos la matriz.
        desc_column = votos[:, col]
        # Obtenemos el indice del mejor valor de semejanza
        min_index = np.argmax(desc_column)
        # Aumentamos en uno el número de votos.
        palabrasImg[min_index] += 1


    # Guardamos palabras y número de votos.
    palabras = [[index, palabrasImg[index]] for index in range(len(palabrasImg)) if palabrasImg[index] > 0]
    palabras = dict(palabras)


    return [palabras,palabrasImg]

# =============================================================================
# Crea el indice invertido + bolsa de palabras dado un diccionario
# =============================================================================
def crearIndicesInvertidos(diccionario):
    #Cargamos los nombre de todas las imagenes
    nombres = [img for img in listdir('./') if isfile(img) and '.pkl' not in img ]
    #Creamos un diccionario que contendra el modelo indice invertido
    indicesInvertidos = dict([index,[]] for index in range(diccionario.shape[0]))
    #Creamos la bolsa de palabras conjunta
    bolsa = dict()

    #Para cada imagen
    for fich in nombres:
        print("calculando votos para:", fich)
        #Leelemos la imagen
        img = leeimagen(fich,1)
        #Calculamos los votos de esta imagen con el diccionario de entrada
        votos, bolsaImg = calcularVotos(img,diccionario)

        #Alamacenamos en la bolsa de palabras final el resultado obtenido
        #para la imagen en la que estamos.
        bolsa[fich] = bolsaImg

        #Recorremos los votos y añadimos para cada uno de los indices,
        #la propia imagen.
        for index,vot in votos.items():
            indicesInvertidos[index].append(fich)

    #Devolvemos el modelo de indice invertido mas la bolsa de palabras
    return [indicesInvertidos, bolsa]

# =============================================================================
# Funcion para obtener imagenes semejantes mediante las palabras de una imagen
# =============================================================================
def imagenesMismasPalabras(diccionario,img, indicesInvertidos):
    # Calculamos los votos de la imágenes.
    votos, bolsaImg = calcularVotos(img,diccionario)

    # Obtenemos las imágenes que también tienen esas palabras.
    imagenes = set()
    for palabra, vot in votos.items():
        for nombre in indicesInvertidos[palabra]:
            imagenes.add(nombre)

    return imagenes

# =============================================================================
# Funcion que calcula las imagens similares a una imagen dada utilizando
# un modelo de indice invertido + bolsa de palabras
# =============================================================================
def calcularImagenesSimilares(img,indicesInvertidos,diccionario, bolsa,nmax=5):
    # Calculamos las imágenes que tienen los mismas palabras que la imagen.
    imgs = imagenesMismasPalabras(diccionario,img,indicesInvertidos)

    # calculamos la bolsa de palabras de las imagenes que tienen
    # palabras parecidas con las palabras de la imagen.
    bolsaImgs = dict([fich,bolsa[fich]] for fich in imgs )
    # Calculamos la bolsa de palabras de nuestra imagen.
    vot, bolsaImg = calcularVotos(img,diccionario)


    # Para cada imagen de bolsaImgs calculamos la semejanza con nuestra imagen.

    similar = [ [fich, ( np.dot(bolsaImg,bImg.T) ) / (norm(bolsaImg)*norm(bImg))]
                            for fich,bImg in bolsaImgs.items()]

    similar = sorted(similar, key=lambda par: par[1], reverse=True)[:nmax+1]

    similar = dict(similar)

    return similar

# =============================================================================
# Funcion que calcula los descriptores mas cercanos a la palabra visual dada
# =============================================================================
def descriptoresCercanos(descriptores,palabra):

    # Calculamos las distancias entre la palabra y el descriptor.
    distancias = [[index,cv2.norm(src1=palabra,src2=desc, normType=cv2.NORM_L2)] for index,desc in descriptores.items()]

    # Ordenamos por menor distancia y nos quedamos con los 10 primeros.
    distancias = sorted(distancias, key=lambda d: d[1])[:10]

    distancias = dict(distancias)

    # Devolvemos los 10 descriptores más cercanos.
    return distancias

# =============================================================================
# Funcion que calcula los descriptores dada una palabra
# =============================================================================
def descriptoresPalabra(i_palabra,descriptores,etiquetas):
    # Obtenemos los índices de los descriptores.
    i_desc = [index for index in range(len(etiquetas)) if etiquetas[index] == i_palabra]

    # Guardamos el subconjunto en un diccionario.
    descriptoresPalabra = dict([index, descriptores[index]] for index in i_desc)

    # Devolvemos el subconjunto.
    return descriptoresPalabra

# =============================================================================
# Funcion que calcula los parches mas cercanos dada una palabra visual
# =============================================================================
def calcularMejoresParches(i_palabra,vocabulario,descriptores,etiquetas,parches):
    # calculamos los parches de la palabra.
    descriptores_sub = descriptoresPalabra(i_palabra, descriptores, etiquetas)
    mejores_desc = descriptoresCercanos(descriptores_sub, vocabulario[i_palabra])
    mejores = dict([[ID,parches[ID]] for ID in mejores_desc.keys()])

    # devolvemos los parches
    return mejores

def mismaEscena(imgs,m_invertido,diccionario,bolsaImagenes):
    bolsa = dict([fich,bolsaImagenes[fich]] for fich in imgs )
    
    bolsaImg1 = bolsa.get(imgs[0])
    
    sim = [ (( np.dot(bolsaImg1,bimg.T) ) / (norm(bolsaImg1)*norm(bimg)))*255 for name,bimg in bolsa.items()]
    
    sim = np.array(sim,dtype=np.uint8).reshape((1,5))

    return sim

###############################################################################
###############################################################################
###############################################################################
###############################################################################

# =============================================================================
# 1. Emparejamiento de descriptores [4 puntos]
# - Mirar las imágenes en imagenesIR.rar y elegir parejas de imágenes
#     que tengan partes de escena comunes. Haciendo uso de una máscara
#     binaria o de las funciones extractRegion() y clickAndDraw(), se-
#     leccionar una región en la primera imagen que esté presente en la se-
#     gunda imagen. Para ello solo hay que fijar los vértices de un polıgono
#     que contenga a la región.
# - Extraiga los puntos SIFT contenidos en la región seleccionada de la
#     primera imagen y calcule las correspondencias con todos los puntos
#     SIFT de la segunda imagen (ayuda: use el concepto de máscara con
#     el parámetro mask).
# - Pinte las correspondencias encontrados sobre las imágenes.
# - Jugar con distintas parejas de imágenes, valorar las correspondencias
#     correctas obtenidas y extraer conclusiones respecto a la utilidad de
#     esta aproximación de recuperación de regiones/objetos de interés a
#     partir de descriptores de una región.
# =============================================================================

# =============================================================================
# EJEMPLO 1 - img95 - img116
# =============================================================================

# Leo las dos imágenes seleccionadas.
img1 = leeimagen('95.png',1);
img2 = leeimagen('116.png',1);

#Obtenemos los vertices correspondientes a una zona de la imagen1
#vertices = extractRegion(img1);
vertices = [(335, 53), (414, 52), (412, 154), (326, 144), (335, 53)]

#Creamos la máscara
mask = crearMascara(img1,vertices)

#Muestro la máscara
multIM([mask],1,1,10,10,False)
input()

#Creamos las correspondencias y mostramos el resultado
res = calcularCorrespondencias(img1,img2,mask)
   
multIM([res],1,1,15,15)

input()

# =============================================================================
# EJEMPLO 2 - img130 - img128
# =============================================================================

# Leo las dos imágenes seleccionadas.
img1 = leeimagen('130.png',1);
img2 = leeimagen('128.png',1);

#Obtenemos los vertices correspondientes a una zona de la imagen1
#vertices = extractRegion(img1);
vertices = [(230, 240),
 (234, 413),
 (356, 452),
 (497, 366),
 (489, 220),
 (386, 198),
 (230, 240)]

#Creamos la máscara
mask = crearMascara(img1,vertices)

#Muestro la máscara
multIM([mask],1,1,10,10,False)
input()

#Creamos las correspondencias y mostramos el resultado
res = calcularCorrespondencias(img1,img2,mask)
   
multIM([res],1,1,15,15)    

input()

# =============================================================================
# EJEMPLO 3 - img234 - img248
# =============================================================================

# Leo las dos imágenes seleccionadas.
img1 = leeimagen('234.png',1);
img2 = leeimagen('248.png',1);

#Obtenemos los vertices correspondientes a una zona de la imagen1
#vertices = extractRegion(img1);
vertices = [(165, 67), (209, 61), (220, 158), (166, 157), (165, 67)]

#Creamos la máscara
mask = crearMascara(img1,vertices)

#Muestro la máscara
multIM([mask],1,1,10,10,False)
input()

#Creamos las correspondencias y mostramos el resultado
res = calcularCorrespondencias(img1,img2,mask)
   
multIM([res],1,1,15,15)

input()

# =============================================================================
# 2. Recuperación de imágenes [4 puntos]
# - Implementar un modelo de ındice invertido + bolsa de palabras para
#     las imágenes dadas en imagenesIR.rar usando el vocabulario dado
#     en kmeanscenters2000.pkl.
# - Verificar que el modelo construido para cada imagen permite recu-
#     perar imágenes de la misma escena cuando la comparamos al resto
#     de imágenes de la base de datos.
# - Elegir dos imágenes-pregunta en las se ponga de manifiesto que el
#     modelo usado es realmente muy efectivo para extraer sus semejantes y
#     elegir otra imagen-pregunta en la que se muestre que el modelo puede
#     realmente fallar. Para ello muestre las cinco imágenes más semejantes
#     de cada una de las imágenes-pregunta seleccionadas usando como
#     medida de distancia el producto escalar normalizado de sus vectores
#     de bolsa de palabras.
# - Explicar qué conclusiones obtiene de este experimento.
# =============================================================================


#Cargamos el diccionario
accu, etiquetas, vocabulario = loadDictionary("kmeanscenters2000.pkl")

#Creamos el modelo de indice invertido + bolsa de palabras
modeloInvertido, bolsa = crearIndicesInvertidos(vocabulario)

# =============================================================================
# VERIFICAR MISMA ESCENA
# =============================================================================

imgs = ['106.png','107.png','108.png','109.png','25.png']

escena = mismaEscena(imgs,modeloInvertido,vocabulario,bolsa)

multIM([escena],1,1,15,15,False)

input()

# =============================================================================
# EJEMPLO 1 - img425
# =============================================================================
#Leo la imagen
img1 = leeimagen('425.png',1)

#Calculamos las imagenes similares a la imagen cargada anteriormente utilizando
#el diccionario anterior.
similar = calcularImagenesSimilares(img1,modeloInvertido,vocabulario,bolsa)

#Preparo la visualizacion
imgs = [leeimagen(fich,1) for fich,sim in similar.items()]

#Muestro el resultado
multIM(imgs,3,3,15,15)

input()

# =============================================================================
# EJEMPLO 2 - img104
# =============================================================================
#Leo la imagen
img1 = leeimagen('104.png',1)

#Calculamos las imagenes similares a la imagen cargada anteriormente utilizando
#el diccionario anterior.
similar = calcularImagenesSimilares(img1,modeloInvertido,vocabulario,bolsa)

#Preparo la visualizacion
imgs = [leeimagen(fich,1) for fich,sim in similar.items()]

#Muestro el resultado
multIM(imgs,3,3,15,15)

input()

# =============================================================================
# EJEMPLO 3 - img62
# =============================================================================
#Leo la imagen
img1 = leeimagen('62.png',1)

#Calculamos las imagenes similares a la imagen cargada anteriormente utilizando
#el diccionario anterior.
similar = calcularImagenesSimilares(img1,modeloInvertido,vocabulario,bolsa)

#Preparo la visualizacion
imgs = [leeimagen(fich,1) for fich,sim in similar.items()]

#Muestro el resultado
multIM(imgs,3,3,15,15)

input()


# =============================================================================
# 3. Visualización del vocabulario [3 puntos]
# - Usando las imágenes dadas en imagenesIR.rar se han extraido 600
#     regiones de cada imagen de forma directa y se han re-escalado en
#     parches de 24x24 pıxeles. A partir de ellas se ha construido un vocab-
#     ulario de 2.000 palabras usando k-means. Los ficheros con los datos
#     son descriptorsAndpatches2000.pkl (descriptores de las regiones
#     y los parches extraıdos) y kmeanscenters2000.pkl (vocabulario ex-
#     traıdo).
# - Elegir al menos dos palabras visuales diferentes y visualizar las re-
#     giones imagen de los 10 parches más cercanos de cada palabra visual,
#     de forma que se muestre el contenido visual que codifican (mejor en
#     niveles de gris).
# - Explicar si lo que se ha obtenido es realmente lo esperado en términos
#     de cercanı́a visual de los parches.
# =============================================================================


# =============================================================================
# EJEMPLO 1 - PALABRA 4
# =============================================================================
#Cargamos el fichero de parches y descriptores
desc, patch = loadAux('descriptorsAndpatches2000.pkl',flagPatches=True)

#Calculamos los mejores parches para una imagen visual determinada
mejores_parches = calcularMejoresParches(4,vocabulario,desc,etiquetas,patch)

#Preparo la visualizacion (en grises porque lo especifica el enunciado)
imgs = [cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) for ID,img in mejores_parches.items()]

#Muestro el resultado
multIM(imgs,5,5,15,15,False)

input()

# =============================================================================
# EJEMPLO 2 - PALABRA 18
# =============================================================================
#Cargamos el fichero de parches y descriptores
mejores_parches = calcularMejoresParches(18,vocabulario,desc,etiquetas,patch)

#Preparo la visualizacion (en grises porque lo especifica el enunciado)
imgs = [cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) for ID,img in mejores_parches.items()]

#Muestro el resultado
multIM(imgs,5,5,15,15,False)

input()

# =============================================================================
# EJEMPLO 2 - PALABRA 21
# =============================================================================
#Cargamos el fichero de parches y descriptores
mejores_parches = calcularMejoresParches(21,vocabulario,desc,etiquetas,patch)

#Preparo la visualizacion (en grises porque lo especifica el enunciado)
imgs = [cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) for ID,img in mejores_parches.items()]

#Muestro el resultado
multIM(imgs,5,5,15,15,False)

input()

