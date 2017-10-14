
# coding: utf-8

# In[4]:

import numpy as np 
a = np.array([1, 2, 3])
print (type(a))
print (a.shape) 
print (a[0], a[1], a[2]) 
a[0] = 5 
print (a) 
b = np.array([[1,2,3],[4,5,6]])  
print (b.shape) 
print (b[0, 0], b[0, 1], b[1, 0])


# In[22]:

a = np.zeros((2,2))
print(a)

b = np.ones((1,2))
print (b) 

c = np.full((2,2), 7, int)
print(c)

d = np.eye(2) 
print (d) 

e = np.random.random((2,2)) 
print(e)


# In[30]:

a = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]])
b = a[:2, 1:3]

print(a)
print()
print(b)

print (a[0, 1] )

b[0, 0] = 77
print (a[0, 1] )


# In[37]:

a = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]])

row_r1 = a[1, :]
row_r2 = a[1:2, :]
print (row_r1, row_r1.shape)
print (row_r2, row_r2.shape)
print()

col_r1 = a[:, 1]
col_r2 = a[:, 1:2]
print (col_r1, col_r1.shape)
print (col_r2, col_r2.shape)


# In[33]:

a = np.array([[1,2], [3, 4], [5, 6]])

print (a[[0, 1, 2], [0, 1, 0]]) 
print (np.array([a[0, 0], a[1, 1], a[2, 0]]))
print (a[[0, 0], [1, 1]])
print (np.array([a[0, 1], a[0, 1]]))


# In[36]:

a = np.array([[1,2,3], [4,5,6], [7,8,9], [10, 11, 12]])

print (a)
print()
b = np.array([0, 2, 0, 1])
print (a[np.arange(4), b])
print()
a[np.arange(4), b] += 10
print (a)


# In[39]:

a = np.array([[1,2], [3, 4], [5, 6]])

bool_idx = (a > 2)
print (bool_idx)
print (a[bool_idx])
print (a[a > 2])


# In[41]:

x = np.array([1, 2])
print (x.dtype)
x = np.array([1.0, 2.0])
print (x.dtype)  
x = np.array([1, 2], dtype=np.int64) 
print (x.dtype)


# In[44]:

x = np.array([[1,2],[3,4]], dtype=np.float64) 
y = np.array([[5,6],[7,8]], dtype=np.float64)

print (x + y)
print (np.add(x, y))
print()

print (x - y)
print (np.subtract(x, y))
print()

print (x * y)
print (np.multiply(x, y))
print()

print (x / y)
print (np.divide(x, y))
print()

print (np.sqrt(x))


# In[48]:

x = np.array([[1,2],[3,4]])
y = np.array([[5,6],[7,8]])
v = np.array([9,10])
w = np.array([11, 12])

print (v.dot(w))
print (np.dot(v, w))
print()
print (x.dot(v))
print (np.dot(x, v))
print()
print (x.dot(y))
print (np.dot(x, y))


# In[49]:

x = np.array([[1,2],[3,4]])

print (np.sum(x))
print (np.sum(x, axis=0))
print (np.sum(x, axis=1))


# In[52]:

x = np.array([[1,2], [3,4]])
print(x)
print(x.T)
print()
v = np.array([1,2,3])
print(v)
print(v.T)


# In[53]:

x = np.array([[1,2,3], [4,5,6], [7,8,9], [10, 11, 12]])
v = np.array([1, 0, 1])
y = np.empty_like(x)

for i in range(4):
    y[i, :] = x[i, :] + v
print(y)


# In[54]:

x = np.array([[1,2,3], [4,5,6], [7,8,9], [10, 11, 12]])
v = np.array([1, 0, 1])
vv = np.tile(v, (4, 1))
print(vv)

y = x + vv
print(y)


# In[56]:

x = np.array([[1,2,3], [4,5,6], [7,8,9], [10, 11, 12]])
v = np.array([1, 0, 1])
y = x + v
print(y)


# In[60]:

v = np.array([1,2,3])
w = np.array([4,5]) 
print (np.reshape(v, (3, 1)) * w)
print()

x = np.array([[1,2,3], [4,5,6]])
print (x + v)
print()
print ((x.T + w).T)
print()
print (x + np.reshape(w, (2, 1)))
print()
print (x * 2)


# In[103]:

from scipy.misc import imread, imsave, imresize

img = imread('cat.jpg')
print (img.dtype, img.shape)

img_tinted = img * [1, 0.95, 0.9]
img_tinted = imresize(img_tinted, (300, 300))

imsave('assets/cat_tinted.jpg', img_tinted)


# In[79]:

from scipy.spatial.distance import pdist, squareform

x = np.array([[0, 1], [1, 0], [2, 0]])
print(x)

d = squareform(pdist(x, 'euclidean'))
print(d)


# In[81]:

import matplotlib.pyplot as plt

x = np.arange(0, 3 * np.pi, 0.1)
y = np.sin(x)

plt.plot(x, y)
plt.show()


# In[82]:

x = np.arange(0, 3 * np.pi, 0.1)
y_sin = np.sin(x)
y_cos = np.cos(x)

plt.plot(x, y_sin)
plt.plot(x, y_cos)
plt.xlabel('x axis label')
plt.ylabel('y axis label')
plt.title('Sine and Cosine')
plt.legend(['Sine', 'Cosine'])
plt.show()


# In[83]:

x = np.arange(0, 3 * np.pi, 0.1)
y_sin = np.sin(x)
y_cos = np.cos(x)

plt.subplot(2, 1, 1)

plt.plot(x, y_sin)
plt.title('Sine')

plt.subplot(2, 1, 2)
plt.plot(x, y_cos)
plt.title('Cosine')
plt.show()


# In[104]:

img = imread('cat.jpg')
img_tinted = img * [1, 0.95, 0.9]

plt.subplot(1, 2, 1)
plt.imshow(img)

plt.subplot(1, 2, 2)

plt.imshow(np.uint8(img_tinted))
plt.show()

