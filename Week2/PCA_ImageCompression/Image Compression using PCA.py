#!/usr/bin/env python
# coding: utf-8

# # Image compression using PCA

# We will see how to use PCA to do image compression!
# 
# We will use the "Labeled Faces in the Wild Home" dataset from the University of Massachusetts Amherst at the following link: http://vis-www.cs.umass.edu/lfw/. It has a horde of pictures originally meant for face recognition. We will use it for image compression illustration purposes here. 

# Library Imports

# In[1]:


from sklearn.datasets import fetch_lfw_people
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


# Fetch the dataset (~300MB)
# it would take a lot of time the first time you run this based on your internet connection speed

# In[2]:


lfw=fetch_lfw_people() 


# Print the attributes of the image dataset. The attributes are saying that there are 13233 images of size 62 x 47 (2914) pixels 

# In[3]:


print(lfw.keys())

_, h, w = lfw.images.shape
print(lfw.images.shape)


# Method for plotting the images

# In[4]:


def plot_gallery(images, h, w, rows=3, cols=4):
    plt.figure()
    for i in range(rows * cols):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(images[i].reshape((h, w)), cmap=plt.cm.bone)
        plt.xticks(())
        plt.yticks(())


# In[5]:


plot_gallery(lfw.data, h, w)


# ## Perform PCA / compression
# 
# Let us see how much variance on the picture is retained if we compressed these down from 62 x 47 to 12 x 10 pixel images

# In[6]:


X,y=lfw.data, lfw.target 
pca_lfw = PCA(120) 
X_proj = pca_lfw.fit_transform(X) 
print(X_proj.shape)


# In[7]:


print(np.cumsum(pca_lfw.explained_variance_ratio_))


# ### Quite impressive! We are able to retain about 92% variance on the original images if we compressed them down to 12 x 10 pixels from an original image size of 62 x 47 pixels
# 
# 
# The parameter `components_` of the `estimator` object gives the components with maximum variance. Below we'll try to visualize the top few principal components. This is just an eigenvector representation, NOT a reconstruction of the original data. In other words We are just visualizing the principal components as images. The principal components are vectors of the length = to the number of features 2914. We'll need to reshape it to a 62 x 47 matrix.

# In[8]:


plot_gallery(pca_lfw.components_, 62, 47, 2, 6)


# ### Magnificient! let's now try to reconstruct the images using the new compressed dataset. 
# 
# In other words, we transformed the #62x47 pixel images into 12x10 images. Now to visualize how these images look we need to inverse transform the 12x10 images back to 62x47 dimension. Note that we're not reverting back to the original data, we're simply going back to the actual dimension of the original images so that we can visualize them. 

# In[9]:


X_inv_proj = pca_lfw.inverse_transform(X_proj) 

#reshaping as 13233 images of 62x47 dimension 
X_proj_img = np.reshape(X_inv_proj,(13233,62,47)) 


# In[10]:


plot_gallery(X_proj_img, h, w)


# ### Not bad at all!
# 
# 

# 
# 
# # Image compression on a natural picture
# 
# Let's try out a different example and save the result to compare the sizes. 
# Let's import this image called "Scenery.jpg", stored in the same working directory as this notebook
# The original image size is 5767 KB

# In[11]:


import matplotlib.image as mi 
img = mi.imread('Scenery.jpg') 

#Now, let's look at the size of this numpy array object img as well as plot it using imshow. 
print(img.shape)
plt.axis('off') 
plt.imshow(img)


# Okay, so the array has 2961 rows each of pixels 8192x3. Let's reshape it into a format that PCA can understand. 
# 
# 8192 * 3 = 24576

# In[12]:


img_rs = np.reshape(img, (2961, 24576)) 
print(img_rs.shape) 


# Great, now lets run PCA with 256 components (16x16 pixels) and transform the image.

# In[13]:


impca = PCA(256).fit(img_rs) 
img_cmp = impca.transform(img_rs) 
print(img_cmp.shape) 
print(np.sum(impca.explained_variance_ratio_)) 


# Awesome! looks like with 256 components we can explain about 95% of the variance. Now to visualize how PCA has performed this compression, let's inverse transform the PCA output and reshape for visualization using imshow. 

# In[14]:


temp = impca.inverse_transform(img_cmp) 
print(temp.shape)

#reshaping 24576 back to the original 8192 * 3 
temp = np.reshape(temp, (2961,8192,3)) 
print(temp.shape)


# Great! now lets visualize like before with imshow and save the new picture to disk to check its size

# In[15]:


plt.axis('off') 
plt.imshow((temp / 255))
plt.savefig('Compressed_Scenery.jpg', bbox_inches='tight')


# The resulting picture when saved on disk as "Compressed_Scenery.jpg" is a meagre 33 KB sized file. That is about 175 times smaller than the original image. We have used python Scikit-learn's PCA library out of the box. We could as well write this PCA algorithm from scratch and use that! 
