#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: jose
"""

import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import random

#Cambiamos el directorio de trabajo
os.chdir('./imagenes/')

# =============================================================================
# Funcion para leer una imagen
# flagColor: 0 Grises, 1 Color
# =============================================================================
def leeimagen(filename,flagColor):
    #Cargamos la imagen
    img = cv2.imread(filename,flagColor)
    
    return img

# =============================================================================
#  Funcion para mostrar una imagen.
# =============================================================================
def pintaI(im):
    #Mostramos la imagen
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
#    img = (np.clip(im,0,1)*255.).astype(np.uint8)
    cv2.imshow('image',im)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

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
# Funcion para mostrar los puntos por octavas
#  Funcion del ejercicio 1.b
# =============================================================================
def pintaPuntos(img,kp,colores,surf=True):
    #Hago una copia para no modificar la imagen que se nos pasa
    img2 = img.copy()
    
    #Para cada keyPoint
    for p in kp:
        #Obtenemos su octava
        if(surf):
            oc = p.octave
        else:
            oc,l,s = unpackSIFTOctave(p)
            oc += 1
        #Dibujamos la circunferencia en la imagen
        cv2.circle(img2,(int(p.pt[0]),int(p.pt[1])),int(p.size/2),list(map(int,colores[oc])),1,cv2.LINE_AA )
    
    return img2

# =============================================================================
# Nos devuelve un vector indicadno el numero de puntos que hay por octava
#  Funcion del ejercicio 1.b
# =============================================================================
def puntosPorOctava(kp,surf=True):
    #Inicializamos un vector con todo ceros
    puntos = np.zeros(20)
    
    #Por cada keyPoint
    for p in kp:
        #Obtenemos su octava
        if surf:
            oc = p.octave
        else:
            oc,l,s = unpackSIFTOctave(p)
            oc+=1
        
        #Sumamos uno a la posicion correspondiente a la octava
        puntos[oc] += 1
        
    return puntos[puntos>0]

# =============================================================================
# Nos devuelve un vector indicadno el numero de puntos que hay por capa
#  Funcion del ejercicio 1.b    
# =============================================================================
def puntosPorCapa(kp,octava):
    #Inicializamos un vector a 0
    puntos = np.zeros(20)
    
    #Por cada keyPoint
    for p in kp:
        #Obtenemos su octava
        oc,l,s = unpackSIFTOctave(p)
        oc+=1
        #Si la octava coincide con la que queremos calcular sumamos uno a la
        # posicion correspondiente en el vector
        if oc == octava:
            puntos[l-1] += 1
        
    return puntos[puntos>0]

# =============================================================================
#  Funcion para desempaquetar los keypoints del método SIFT
# =============================================================================
def unpackSIFTOctave(kpt):
    """unpackSIFTOctave(kpt)->(octave,layer,scale)
    @created by Silencer at 2018.01.23 11:12:30 CST
    @brief Unpack Sift Keypoint by Silencer
    @param kpt: cv2.KeyPoint (of SIFT)
    """
    _octave = kpt.octave
    octave = _octave&0xFF
    layer  = (_octave>>8)&0xFF
    if octave>=128:
        octave |= -128
    if octave>=0:
        scale = float(1/(1<<octave))
    else:
        scale = float(1<<-octave)
    return (octave, layer, scale)
    
# =============================================================================
#  Funcion que elimina los bordes negros de una imagen
#  Funcion para el ejercicio 3
# =============================================================================
def recortar(img):
    #Pasamos la imagen a gris
    gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    x,y,z,t = cv2.boundingRect(gray)
    #Obtenemos la imagen quitando lo que sobra
    new = img[y:y+t,x:x+z]
    return new

# =============================================================================
# Funcion que permite crear un mosaico a partir de 3 imagenes.
# Funcion para el ejercicio 3
# =============================================================================
def mosaico3(imgs,kps,matches,r=True):
    maxx = 0
    maxy = 0
    #Obtenemos el tamaño de la imagen mas grande
    for im in imgs:
        maxy = im.shape[0] if im.shape[0] > maxy else maxy
        maxx = im.shape[1] if im.shape[1] > maxx else maxx
    
    #Calculamos el tamaño del lienzo final
    mosaico_size = [maxx*3,maxy*3]
    
    #Creo la matriz de traslacion
    H = np.eye(3)
    H[1][2] = (mosaico_size[1]/2-(maxy/2))
    
    #Posicionamos la primera imagen en el lienzo
    result = cv2.warpPerspective(imgs[0],H,(mosaico_size[0],mosaico_size[1]))
    
    #Calculamos los puntos necesarios para crear la homografia entre la primera
    #imagen y la siguiente
    p1 = np.array([kps[0][m.queryIdx].pt for m in matches[0]])
    p2 = np.array([kps[1][m.trainIdx].pt for m in matches[0]])
    #Calculamos la homografia entre estas dos imagenes
    homography,masc = cv2.findHomography(p2,p1, cv2.RANSAC,1)
    #Colocamos la imagen en el lienzo junto a al primera
    cv2.warpPerspective(imgs[1],H.dot(homography),(mosaico_size[0],mosaico_size[1]),dst=result,borderMode=cv2.BORDER_TRANSPARENT)
    
    #Calculamos los puntos correspondientes entre la segunda imagen y
    # la tercera
    p1 = np.array([kps[1][m.queryIdx].pt for m in matches[1]])
    p2 = np.array([kps[2][m.trainIdx].pt for m in matches[1]])
    #Calculamos la homografía acumulada
    H = H.dot(homography)
    #Calculamos la homografia entre estas dos imagenes
    homography,masc = cv2.findHomography(p2,p1, cv2.RANSAC,1)
    #Añadimos esta tercera imagen al lienzo final
    cv2.warpPerspective(imgs[2],H.dot(homography),(mosaico_size[0],mosaico_size[1]),dst=result,borderMode=cv2.BORDER_TRANSPARENT)
    
    if(r):
        result = recortar(result)
    
    return result

# =============================================================================
# Funcion que crea un panorama dadas N imagenes y sus respectivos
# KeyPoints con sus matches.
# Funcion ejercicio 3
# =============================================================================
def mosaicoN(imgs,kps,matches,r=True):
    maxx = 0
    maxy = 0
    #Obtenemos el tamaño de la imagen mas grande
    for im in imgs:
        maxy = im.shape[0] if im.shape[0] > maxy else maxy
        maxx = im.shape[1] if im.shape[1] > maxx else maxx
    
    #Calculamos el tamaño del lienzo final
    mosaico_size = [maxx*len(imgs),maxy*len(imgs)]
    
    #Creo la matriz de traslacion
    H = np.eye(3)
    H[1][2] = (mosaico_size[1]/2-(maxy/2))
    
    #Posicionamos la primera imagen en el lienzo
    result = cv2.warpPerspective(imgs[0],H,(mosaico_size[0],mosaico_size[1]))
    
    #Para cada una de las siguientes imágenes, se obtienen los puntos y junto
    # con la de la anterior, se crean las homografías que van acumulandose
    # con las anteriores y voy colocando cada una de las imagenes en el lienzo
    # final
    for i in range(len(imgs)-1):
        p1 = np.array([kps[i][m.queryIdx].pt for m in matches[i]])
        p2 = np.array([kps[i+1][m.trainIdx].pt for m in matches[i]])
        homography,masc = cv2.findHomography(p2,p1, cv2.RANSAC,1)
        cv2.warpPerspective(imgs[i+1],H.dot(homography),(mosaico_size[0],mosaico_size[1]),dst=result,borderMode=cv2.BORDER_TRANSPARENT)
        H = H.dot(homography)
    
    if(r):
        result = recortar(result)
    
    return result

# =============================================================================
# BFMatcher con knn2 y utilizando Lowe-Average
# Funcion para el ejercicio 3 y 4
# =============================================================================
def BFL2NN(des1,des2):
    bf = cv2.BFMatcher(crossCheck=False)

    matches = bf.knnMatch(des1,des2,k=2)

    good = []
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            good.append([m])
        
    return list(map(lambda x:x[0],good))
    
###############################################################################
###############################################################################
###############################################################################
###############################################################################


# =============================================================================
# Ejercicio 1
# (3 puntos) Detección de puntos SIFT y SURF. Aplicar la detección de
# puntos SIFT y SURF sobre las imágenes, representar dichos puntos sobre
# las imágenes haciendo uso de la función drawKeyPoints. Presentar los
# resultados con las imágenes Yosemite.rar.
# =============================================================================
    
img1 = leeimagen("Yosemite1.jpg",1)
gray1 = leeimagen("Yosemite1.jpg",0)

img2 = leeimagen("Yosemite2.jpg",1)
gray2 = leeimagen("Yosemite2.jpg",0)

# SIFT

sift = cv2.xfeatures2d.SIFT_create()
kpsi1 = sift.detect(gray1,None)
kpsi2 = sift.detect(gray2,None)

imgsi1 = cv2.drawKeypoints(gray1,kpsi1,None)
imgsi2 = cv2.drawKeypoints(gray2,kpsi2,None)

# SURF

surf = cv2.xfeatures2d.SURF_create()
kpsf1 = surf.detect(gray1,None)
kpsf2 = surf.detect(gray2,None)

imgsf1 = cv2.drawKeypoints(gray1,kpsf1,None)
imgsf2 = cv2.drawKeypoints(gray2,kpsf2,None)

multIM([imgsi1,imgsi2,imgsf1,imgsf2],2,2,15,10)

input()

# =============================================================================
# Ejercicio 1 - Apartado (a)
# (a) Variar los valores de umbral de la función de detección de puntos
# hasta obtener un conjunto numeroso (≥ 1000) de puntos SIFT y
# SURF que sea representativo de la imagen. Justificar la elección
# de los parámetros en relación a la representatividad de los puntos
# obtenidos.
# =============================================================================


# SIFT

sift = cv2.xfeatures2d.SIFT_create(contrastThreshold=0.06,edgeThreshold=6)
kpsi1 = sift.detect(gray1,None)
kpsi2 = sift.detect(gray2,None)

imgsi1 = cv2.drawKeypoints(gray1,kpsi1,None)
imgsi2 = cv2.drawKeypoints(gray2,kpsi2,None)

# SURF

surf = cv2.xfeatures2d.SURF_create(hessianThreshold=400)
kpsf1 = surf.detect(gray1,None)
kpsf2 = surf.detect(gray2,None)

imgsf1 = cv2.drawKeypoints(gray1,kpsf1,None)
imgsf2 = cv2.drawKeypoints(gray2,kpsf2,None)

multIM([imgsi1,imgsi2,imgsf1,imgsf2],2,2,15,10)

input()

# =============================================================================
# Ejercicio 1 - Apartado (b)
# (b) Identificar cuántos puntos se han detectado dentro de cada octava.
# En el caso de SIFT, identificar también los puntos detectados en
# cada capa. Mostrar el resultado dibujando sobre la imagen original
# un cı́rculo centrado en cada punto y de radio proporcional al valor de
# sigma usado para su detección (ver circle()) y pintar cada octava
# en un color.
# =============================================================================

# SIFT

siftm = cv2.xfeatures2d.SIFT_create(1000)
kpsi1m = siftm.detect(gray1,None)
kpsi2m = siftm.detect(gray2,None)

print("La imagen 1 con SIFT tiene "+str(len(kpsi1m))+" KeyPoints.")
pp = puntosPorOctava(kpsi1m,False)
for i,p in enumerate(pp):
    print("La octava "+str(i)+" tiene "+str(int(pp[i]))+" KeyPoints.")
    cp = puntosPorCapa(kpsi1m,i)
    for j in range(cp.size):
        print("La capa "+str(j)+" de la octava "+str(i)+" tiene "+str(int(cp[j]))+" KeyPoints.")
    
print("\nLa imagen 2 con SIFT tiene "+str(len(kpsi2m)))
pp = puntosPorOctava(kpsi2m,False)
for i,p in enumerate(pp):
    print("La octava "+str(i)+" tiene "+str(int(pp[i]))+" KeyPoints.")
    cp = puntosPorCapa(kpsi2m,i)
    for j in range(cp.size):
        print("La capa "+str(j)+" de la octava "+str(i)+" tiene "+str(int(cp[j]))+" KeyPoints.")

nimg1 = cv2.cvtColor(gray1,cv2.COLOR_GRAY2BGR)
nimg2 = cv2.cvtColor(gray2,cv2.COLOR_GRAY2BGR)

colores = np.array([[0, 0,  255],
       [  0, 255, 0],
       [ 255, 0, 0],
       [ 255, 0, 255],
       [255,  255,  0],
       [0, 255,  255],
       [255, 0,  125],
       [125, 0,  255],
       [0,0,0],
       [0, 150, 255]], dtype=np.uint8)

imgsi1 = pintaPuntos(nimg1,kpsi1m,colores,False)
imgsi2 = pintaPuntos(nimg2,kpsi2m,colores,False)

# SURF

surfm = cv2.xfeatures2d.SURF_create(2000)
kpsf1m = surfm.detect(gray1,None)
kpsf2m = surfm.detect(gray2,None)

print("\nLa imagen 1 con SURF tiene "+str(len(kpsf1m)))
pp = puntosPorOctava(kpsf1m)
for i,p in enumerate(pp):
    print("La octava "+str(i)+" tiene "+str(int(pp[i]))+" KeyPoints.")
    
print("\nLa imagen 2 con SURF tiene "+str(len(kpsf2m)))
pp = puntosPorOctava(kpsf2m)
for i,p in enumerate(pp):
    print("La octava "+str(i)+" tiene "+str(int(pp[i]))+" KeyPoints.")
    
imgsf1 = pintaPuntos(nimg1,kpsf1m,colores)
imgsf2 = pintaPuntos(nimg2,kpsf2m,colores)

multIM([imgsi1,imgsi2,imgsf1,imgsf2],2,2,15,10)

input()

# =============================================================================
# Ejercicio 1 - Apartado (c)
# (c) Mostrar cómo con el vector de keyPoint extraıdos se pueden calcu-
# lar los descriptores SIFT y SURF asociados a cada punto usando
# OpenCV.
# =============================================================================

#SIFT

kpsi1,desi1 = sift.compute(gray1,kpsi1)
kpsi2,desi2 = sift.compute(gray2,kpsi2)

print("img1 SIFT")
print("kp = "+str(len(kpsi1)))
print("des = "+str(desi1.shape))
print("img2 SIFT")
print("kp = "+str(len(kpsi2)))
print("des = "+str(desi2.shape))

#SURF

kpsf1,desf1 = sift.compute(gray1,kpsf1)
kpsf2,desf2 = sift.compute(gray2,kpsf2)

print("img1 SURF")
print("kp = "+str(len(kpsf1)))
print("des = "+str(desf1.shape))
print("img2 SURF")
print("kp = "+str(len(kpsf2)))
print("des = "+str(desf2.shape))

input()

# =============================================================================
# Ejercicio 2
# 2. (2.5 puntos) Usar el detector-descriptor SIFT de OpenCV sobre las imágenes
# de Yosemite.rar (cv2.xfeatures2d.SIFT create()). Extraer sus lis-
# tas de keyPoints y descriptores asociados. Establecer las corresponden-
# cias existentes entre ellos usando el objeto BFMatcher de OpenCV y los
# criterios de correspondencias “BruteForce+crossCheck y “Lowe-Average-
# 2NN”. (NOTA: Si se usan los resultados propios del puntos anterior en
# lugar del cálculo de SIFT de OpenCV se añaden 0.5 puntos)
# =============================================================================

# =============================================================================
# Ejercicico 2 - Apartado (a)
# (a) Mostrar ambas imágenes en un mismo canvas y pintar lıneas de difer-
# entes colores entre las coordenadas de los puntos en correspondencias.
# Mostrar en cada caso 100 elegidas aleatoriamente.
# =============================================================================

# =============================================================================
#  BruteForce+crossCheck
# =============================================================================

bf = cv2.BFMatcher(crossCheck=True)

matches = bf.match(desi1,desi2)

matches = sorted(matches, key = lambda x:x.distance)
sample = random.sample(matches,100)

bfc = cv2.drawMatches(gray1,kpsi1,gray2,kpsi2,sample,None,flags=2)

# =============================================================================
#  Lowe-Average-2NN
# =============================================================================

bf = cv2.BFMatcher(crossCheck=False)

matches = bf.knnMatch(desi1,desi2,k=2)

good = []
for m,n in matches:
    if m.distance < 0.7*n.distance:
        good.append([m])
        
sample = random.sample(good,100)

bfnn = cv2.drawMatchesKnn(gray1,kpsi1,gray2,kpsi2,sample,None,flags=2)

multIM([bfc,bfnn],2,1,15,10)

input()

# =============================================================================
# Ejercicio 3
# 3. (2.5 puntos) Escribir una función que genere un mosaico de calidad a
# partir de N = 3 imágenes relacionadas por homografıas, sus listas de
# keyPoints calculados de acuerdo al punto anterior y las correspondencias
# encontradas entre dichas listas. Estimar las homografıas entre ellas usando
# la función cv2.findHomography(p1,p2, CV RANSAC,1). Para el mosaico
# será necesario.
# =============================================================================

img1 = leeimagen("yosemite1.jpg",1)
img2 = leeimagen("yosemite2.jpg",1)
img3 = leeimagen("yosemite3.jpg",1)
gray1 = leeimagen("yosemite1.jpg",0)
gray2 = leeimagen("yosemite2.jpg",0)
gray3 = leeimagen("yosemite3.jpg",0)

imgs = [img1,img2,img3]

sift = cv2.xfeatures2d.SIFT_create()
kp1 = sift.detect(gray1,None)
kp2 = sift.detect(gray2,None)
kp3 = sift.detect(gray3,None)

kp1,des1 = sift.compute(gray1,kp1)
kp2,des2 = sift.compute(gray2,kp2)
kp3,des3 = sift.compute(gray3,kp3)

kps = [kp1,kp2,kp3]
 
matches12 = BFL2NN(des1,des2)
matches23 = BFL2NN(des2,des3)
        
matches = [matches12,matches23]

mosaico = mosaico3(imgs,kps,matches)
    
multIM([mosaico],1,1,10,10)
   
input()    

# =============================================================================
# Ejercicio 4
# 4. (2.5 puntos) Lo mismo que en el punto anterior pero para N > 5 (usar las
# imágenes para mosaico).
# =============================================================================

img1 = leeimagen("mosaico002.jpg",1)
img2 = leeimagen("mosaico003.jpg",1)
img3 = leeimagen("mosaico004.jpg",1)
img4 = leeimagen("mosaico005.jpg",1)
img5 = leeimagen("mosaico006.jpg",1)
img6 = leeimagen("mosaico007.jpg",1)
img7 = leeimagen("mosaico008.jpg",1)
img8 = leeimagen("mosaico009.jpg",1)
img9 = leeimagen("mosaico010.jpg",1)
img10 = leeimagen("mosaico011.jpg",1)
gray1 = leeimagen("mosaico002.jpg",0)
gray2 = leeimagen("mosaico003.jpg",0)
gray3 = leeimagen("mosaico004.jpg",0)
gray4 = leeimagen("mosaico005.jpg",0)
gray5 = leeimagen("mosaico006.jpg",0)
gray6 = leeimagen("mosaico007.jpg",0)
gray7 = leeimagen("mosaico008.jpg",0)
gray8 = leeimagen("mosaico009.jpg",0)
gray9 = leeimagen("mosaico010.jpg",0)
gray10 = leeimagen("mosaico011.jpg",0)

imgs = [img1,img2,img3,img4,img5,img6,img7,img8,img9,img10]

sift = cv2.xfeatures2d.SIFT_create()
kp1 = sift.detect(gray1,None)
kp2 = sift.detect(gray2,None)
kp3 = sift.detect(gray3,None)
kp4 = sift.detect(gray4,None)
kp5 = sift.detect(gray5,None)
kp6 = sift.detect(gray6,None)
kp7 = sift.detect(gray7,None)
kp8 = sift.detect(gray8,None)
kp9 = sift.detect(gray9,None)
kp10 = sift.detect(gray10,None)

kp1,des1 = sift.compute(gray1,kp1)
kp2,des2 = sift.compute(gray2,kp2)
kp3,des3 = sift.compute(gray3,kp3)
kp4,des4 = sift.compute(gray4,kp4)
kp5,des5 = sift.compute(gray5,kp5)
kp6,des6 = sift.compute(gray6,kp6)
kp7,des7 = sift.compute(gray7,kp7)
kp8,des8 = sift.compute(gray8,kp8)
kp9,des9 = sift.compute(gray9,kp9)
kp10,des10 = sift.compute(gray10,kp10)

kps = [kp1,kp2,kp3,kp4,kp5,kp6,kp7,kp8,kp9,kp10]
 
matches12 = BFL2NN(des1,des2)
matches23 = BFL2NN(des2,des3)
matches34 = BFL2NN(des3,des4)
matches45 = BFL2NN(des4,des5)
matches56 = BFL2NN(des5,des6)
matches67 = BFL2NN(des6,des7)
matches78 = BFL2NN(des7,des8)
matches89 = BFL2NN(des8,des9)
matches910 = BFL2NN(des9,des10)
        
matches = [matches12,matches23,matches34,matches45,matches56,matches67,matches78]
matches.extend([matches89,matches910])

mosaico = mosaicoN(imgs,kps,matches)

multIM([mosaico],1,1,10,10)