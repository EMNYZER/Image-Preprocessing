import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
import pandas as singgih
import math
# library custom PCD 
# oleh : Muhammad Fahir
def togray(image):
    # kernelgauss = np.ones((3,3))/9
    image1 = image[:,:,2]
    image0 = image[:,:,0]
    image2 = image[:,:,1]
    return image1/3 + image0/3 +image2/3
def addpadd(arr,atas=0,bawah=0,kiri=0,kanan=0,padd="zero"):
    #not yet implement other padd mode
    tinggi,lebar=arr.shape
    arrhasil=np.zeros((tinggi+atas+bawah,lebar+kiri+kanan))
    for i in range(0+atas,tinggi+atas):
        for j in range(0+kiri,lebar+kiri):
            arrhasil[i][j]=arr[i-atas][j-kiri]
    return arrhasil
def translasi(image,x=0,y=0):
    tinggi,lebar=image.shape
    hasil=np.zeros((tinggi,lebar))
    for i in range(tinggi):
        for j in range(lebar):
            x2=i-x
            y2=j-y
            if(y2<0):
                y2=lebar+y2
            if(x2<0):
                x2=tinggi+x2  
            if(y2>=lebar):
                y2=y2-lebar
            if(x2>=tinggi):
                x2=x2-tinggi
            if(0<=x2 <tinggi and 0 <=y2< tinggi):
                hasil[i,j]=image[x2,y2]
    return hasil
def mirror(image,axis="x"):
    tinggi,lebar=image.shape
    hasil=np.zeros((tinggi,lebar))
    if(axis=="x"):
        for i in range(tinggi):
            for j in range(lebar):
                hasil[i,j]=image[tinggi-i-1][j]
        return hasil
    elif(axis=="y"):
        for i in range(tinggi):
            for j in range(lebar):
                hasil[i,j]=image[i][lebar-j-1]
        return hasil
    else:
        print("wrong axis, Use 'x' or 'y' ")
def dilatasi(image,skala):
    tinggi,lebar=image.shape
    hasil=np.zeros((tinggi*skala,lebar*skala))
    for i in range(tinggi*skala):
        for j in range(lebar*skala):
            k=int(i/skala)
            l=int(j/skala)
            hasil[i,j]=image[k,l]
    return hasil
def rotasi(image,derajat=0):
    tinggi,lebar=image.shape
    rad=int(math.sqrt(pow(tinggi/2,2)+pow(lebar/2,2)))
    diameter=rad*2
    hasil=np.zeros((diameter,diameter))
    copy=np.zeros((diameter,diameter))
    plust=int((diameter-tinggi)/2)
    plusl=int((diameter-lebar)/2)
    for i in range(0,tinggi):
        for j in range(0,lebar):
          copy[i+plust][j+plusl]  = image[i,j]
    for i in range(diameter):
        for j in range(diameter):
            x2 = round((i - diameter / 2) * np.cos(np.radians(derajat)) - (j - diameter / 2) * np.sin(np.radians(derajat)) + diameter / 2)
            y2 = round((i - diameter / 2) * np.sin(np.radians(derajat)) + (j - diameter / 2) * np.cos(np.radians(derajat)) + diameter / 2)
            if 0 <= x2 < diameter and 0 <= y2 < diameter:
                hasil[i,j]=copy[x2,y2]
    return hasil
def normalisasi(image,skala_min=0,skala_max=255):
    tinggi,lebar=image.shape
    hasil=np.zeros((tinggi,lebar))
    min=np.min(image)
    max=np.max(image)
    for i in range(tinggi):
        for j in range(lebar):
            hasil[i,j]=round(((image[i,j]-min)*(skala_max-skala_min)/(max-min)) + skala_min -1)
    return hasil
def histvar(image,rangep=256):
    graytindex = [i for i in range(rangep)]
    tinggi,lebar=image.shape
    jumlah = np.zeros(rangep).astype(int)
    for i in range(tinggi):
        for j in range(lebar):
            jumlah[round(image[i][j])]+=1
    return graytindex,jumlah
def fusion(image1,image2,mode=0):
    t1,l1=image1.shape
    t2,l2=image2.shape
    
    if(mode==0):
        hasil=np.zeros((max(t1,t2),l1+l2))
        for i in range(max(t1,t2)):
            for j in range(l1+l2):
                if(j<l1):
                    try:
                        hasil[i,j]=image1[i,j]
                    except:
                        hasil[i,j]=0
                else:
                    try:
                        hasil[i,j]=image2[i,j-l1]
                    except:
                        hasil[i,j]=0
    else:
        hasil=np.zeros((t1+t2,max(l1,l2)))
        for i in range(t1+t2):
            if(i<t1):
                for j in range(max(l1,l2)):
                    try:
                        hasil[i,j]=image1[i,j]
                    except:
                        hasil[i,j]=0
            else:
                for j in range(max(l1,l2)):
                    try:
                        hasil[i,j]=image2[i-t1,j]
                    except:
                        hasil[i,j]=0        
    return hasil
def splitimage(image,ini=1,inj=1):
    listall=[]
    tinggi,lebar=image.shape
    for i in range(ini):
        listin=[]
        for j in range(inj):
            start_row = (tinggi //ini) * i
            end_row = (tinggi // ini) * (i + 1)
            start_col = (lebar // inj) * j
            end_col = (lebar // inj) * (j + 1)
            imagebagi = image[start_row:end_row, start_col:end_col]
            listin.append(imagebagi)
        listall.append(listin)
    return listall
def Syncfusion(listimg):
    iindexing,jindexing=len(listimg),len(listimg[0])
    liststart=[]
    for i in range(iindexing):
        liststart.append(listimg[i][0])
    for i in range(len(liststart)):
        for j in range(1,jindexing):
            liststart[i]=fusion(liststart[i],listimg[i][j])
    if(iindexing>1):
        hasil=fusion(liststart[0],liststart[1],mode=1)
        if(iindexing>2):
            for i in range(2,jindexing):
                hasil=fusion(hasil,liststart[i],mode=1)
    else:
        hasil=liststart[0]
    return hasil
def equalisasi(image):
    tinggi,lebar=image.shape
    jumlah = np.zeros(256).astype(int)
    for i in range(tinggi):
        for j in range(lebar):
            jumlah[int(image[i][j])]+=1
    hist_sum= np.zeros(256).astype(float)
    for i in range(256):
        hist_sum[i]=np.sum(jumlah[0:i+1])
    Target = (hist_sum*255)/(tinggi*lebar)
    Target = np.round(Target).astype(int)
    Hasil = np.zeros(image.shape).astype(int)
    for i in range(tinggi):
        for j in range(lebar):
            Hasil[i][j] = Target[int(image[i][j])]
    return Hasil,Target
def spesifikasi(imageasli,imagetarget):
    tinggi,lebar=imageasli.shape
    _,Target=equalisasi(imageasli)
    _,Target2=equalisasi(imagetarget)
    minAtIndex = np.zeros(256).astype(int)
    for i in range(255):
        min=256
        for j in range(255):
            newmin=abs(Target[i]-Target2[j])
            if min>newmin:
                min=newmin
                minAtIndex[i]=j 
    HasilSpek = np.zeros(imageasli.shape).astype(int)
    for i in range(tinggi):
        for j in range(lebar):
            HasilSpek[i,j]=minAtIndex[round(imageasli[i,j])]
    return HasilSpek
def splitSpec(image,imagegelap,imageterang,iindexing=1,jindexing=1):
    image,_=equalisasi(image)
    listall=splitimage(image,iindexing,jindexing)
    tinggi,lebar=image.shape
    rataratapixel=[]
    for i in range(iindexing):
        rattoappen=[]
        for j in range(jindexing):
            tinggi,lebar=listall[i][j].shape
            rat=0
            for k in range(tinggi):
                for l in range(lebar):
                    rat+=listall[i][j][k][l]
            rat/=tinggi*lebar
            rattoappen.append(rat)
        rataratapixel.append(rattoappen)
    for i in range(iindexing):
        for j in range(jindexing):
            if(rataratapixel[i][j]<=128):
                listall[i][j]=spesifikasi(listall[i][j],imagegelap)
            else:
                listall[i][j]=spesifikasi(listall[i][j],imageterang)
    return Syncfusion(listall)