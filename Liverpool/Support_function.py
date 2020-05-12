#!/usr/bin/env python
# coding: utf-8

# In[11]:


from random import randint, sample
def random_crop(img,w=256,h=256,num=5):
    result=np.ndarray((1,w,h,3),dtype='uint8')
    size=img.size
    width=size[0]
    height=size[1]
    wlist=sample(range((width-w)),num)
    hlist=sample(range((height-h)),num)
    for i in range(num):
        #plt.imshow(img.crop((wlist[i], hlist[i], wlist[i]+256, hlist[i]+256)))
        #img.crop((wlist[i], hlist[i], wlist[i]+256, hlist[i]+256)).show()
        result=np.vstack((result,np.asarray(img.crop((wlist[i], hlist[i], wlist[i]+256, hlist[i]+256)),dtype='uint8')[np.newaxis,]))
    return result[1:]


# In[12]:


from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import numpy as np
def data_generation(dataset,label,w=256,h=256,num=5): #dataset is a one column dataset with a list of img(not matrix)
    label_list=[]
    result=np.ndarray((1,w,h,3),dtype='uint8')
    for i in range(len(dataset)):
        result=np.vstack((result,random_crop(img=dataset.iloc[:,0][i],w=w,h=h,num=num)))
        for j in range(num):
            label_list.append(label[i])
    a=OneHotEncoder()
    onelinedf=pd.DataFrame(label_list)
    label_list=a.fit_transform(onelinedf).toarray()
    return (result[1:],label_list)


# In[13]:


from PIL import Image
img = Image.open('image/Origi1.jpg')
df=pd.DataFrame([img,img])
l=['O','P']
#df.loc[:,0]
data_generation(df,l)[1]


# In[14]:


from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import numpy as np


# In[36]:


a=OneHotEncoder()
onelinedf=pd.DataFrame(['a','b','c','d','d'])
onelinedf
a.fit_transform(onelinedf).toarray()


# In[18]:


def read_picture(name):
    result=[]
    for i in name:
        result.append(Image.open(f'image/{i}.jpg'))
    return result


# In[21]:


Origi=['Origi'+str(i) for i in range(1,8)]
A=['A'+str(i) for i in range(1,8)]
#read_picture(Origi)






