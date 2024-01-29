# -*- coding: utf-8 -*-
"""
Created on Sun May  7 19:05:28 2023

@author: ALIENWARE
"""
import numpy as np
import os
import matplotlib.pyplot as plt
import pickle

def distance_matrix(img):
    n=img.shape[0]*img.shape[1]
    dm=np.zeros((n,n))
    for i in range(0,n):
        for j in range(0,n):
            dm[i,j]=np.linalg.norm(img[i//img.shape[1],i%img.shape[1],:]-img[j//img.shape[1],j%img.shape[1],:])
            #if j%img.shape[1]>(i%img.shape[1])+9 or j%img.shape[1]<(i%img.shape[1])-9 or j//img.shape[1]>(i//img.shape[1])+9 or j//img.shape[1]<(i//img.shape[1])-9:
            #    dm[i,j]=9
    return dm

def weight_matrix(dm,sigma):
    w=np.exp(-np.power(dm,2)/2*(sigma**2))
    n=w.shape[0]
    
    return w

def lap_matrix(w):
    D=np.diag(np.sum(w,axis=1))
    D_1=np.diag(1/np.sqrt(np.sum(w,axis=1)))
    L=D-w
    L_norm=np.dot(np.dot(D_1,L),D_1)
    return L_norm

def sp_cluster(L):
    e_value,e_vector=np.linalg.eig(L)
    e_value=np.real(e_value)
    e_vector=np.real(e_vector)
    e_sb=np.sort(e_value)
    
    x=np.zeros((e_value.shape))
    for i in range(1,len(e_value)):
        x[i]=e_sb[i]-e_sb[i-1]
    
    K=np.argmax(x)
    K=4
    e_se=np.argsort(e_value)
    data_k=np.zeros((L.shape[0],K))
    data_k[:,0]=e_vector[:,e_se[1]]
    for i in range(1,K):
        data_k[:,i]=e_vector[:,e_se[i+1]]#np.hstack((data_k,e_vector[:,e_se[i]]))
    
    from sklearn.cluster import KMeans
    sp_kmeans = KMeans(n_clusters=K).fit(data_k)
    return sp_kmeans.labels_,K

def load_data(data_path, file_name):
    """Load CIFAR-10 data from given data_path"""
    file_path = os.path.join(data_path, file_name)
    with open(file_path, 'rb') as f:
        # load data with pickle
        entry = pickle.load(f, encoding='latin1')
        images = entry['data']
        labels = entry['labels']
        # reshape the array to 32x32 color image
        images = images.reshape(-1, 3, 32, 32)  # '-1' means the value will be deduced from the shape of array
    return images, labels
data_path='./cifar'
file_name='data_batch_1'
images, labels=load_data(data_path, file_name)
#image1=np.zeros((32,32,3))
#image1=np.transpose(images[1,:,:,:],(1,2,0))
#image1[:,:,0]=images[1,0,:,:]
#image1[:,:,1]=images[1,1,:,:]
#image1[:,:,2]=images[1,2,:,:]
#image1=np.uint8(image1*255)
#plt.imshow(image1)
#print(labels[1])
i=1
k=0
while i>0:
    i=i+1
    if labels[i]==7:
        k=k+1
    if k==20 and labels[i]==7:
        image2=np.transpose(images[i,:,:,:],(1,2,0))
        
        i=-1
        break
#image2=np.uint8(image2*255)
#plt.imshow(image2)
image2=image2/255
plt.imshow(image2)
dm=distance_matrix(image2)
w=weight_matrix(dm,0.5)
L_norm=lap_matrix(w)
a,K=sp_cluster(L_norm)
print(a.shape)
image_show=np.copy(image2)
for i in range(0,1024):
    if a[i] ==1:
        image_show[i//32,i%32,:]=[255,0,0]
        #print('yed')
    elif a[i]==2:
        image_show[i//32,i%32,:]=[0,255,0]
    elif a[i]==3:
        image_show[i//32,i%32,:]=[0,0,255]
    else:
        image_show[i//32,i%32,:]=[0,0,0]
print(K)
plt.imshow(image_show)
    
    
        