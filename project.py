import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
import cv2
from scipy.stats import stats
import matplotlib.image as mpimg
from PIL import Image

img = cv2.cvtColor(cv2.imread("avicii-dj-cu-1920x1080.jpg"), cv2.COLOR_BGR2RGB)
plt.imshow(img)
plt.show()
img.shape

#Splitting into channels
blue,green,red = cv2.split(img)

blue_temp_df = pd.DataFrame(data = blue)
blue_temp_df

df_blue = blue/255
df_green = green/255
df_red = red/255

pca_b = PCA(n_components=50)
pca_b.fit(df_blue)
trans_pca_b = pca_b.transform(df_blue)
pca_g = PCA(n_components=50)
pca_g.fit(df_green)
trans_pca_g = pca_g.transform(df_green)
pca_r = PCA(n_components=50)
pca_r.fit(df_red)
trans_pca_r = pca_r.transform(df_red)

b_arr = pca_b.inverse_transform(trans_pca_b)
g_arr = pca_g.inverse_transform(trans_pca_g)
r_arr = pca_r.inverse_transform(trans_pca_r)
#print(b_arr.shape, g_arr.shape, r_arr.shape)

img_reduced= (cv2.merge((b_arr, g_arr, r_arr)))
#print(img_reduced.shape)
fig = plt.figure(figsize = (10, 7.2)) 
fig.add_subplot(121)
plt.title("Original Image")
plt.imshow(img)
fig.add_subplot(122)
plt.title("Reduced Image")
plt.imshow(img_reduced)
plt.show()
