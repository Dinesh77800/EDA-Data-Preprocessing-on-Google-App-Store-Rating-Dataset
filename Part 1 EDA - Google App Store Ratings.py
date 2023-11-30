#!/usr/bin/env python
# coding: utf-8

# 1. Import required libraries and read the dataset.
# 

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
df = pd.read_csv('Apps_data.csv')
df


# 2. Check the first few samples, shape, info of the data and try to familiarize yourself with different features

# In[2]:


# Sample
df.sample(5)


# In[3]:


# shape
df.shape


# In[4]:


# info
df.info()


# 3. Check summary statistics of the dataset. List out the columns that need to be worked upon for model
# building.

# In[5]:


# summmary statistics 
df.describe(include='all')


# In[6]:


a=df.drop(columns='App')
a.columns

# these are the columns needed for model building


# 4. Check if there are any duplicate records in the dataset? if any drop them.
# 

# In[7]:


df.duplicated().sum()


# In[8]:


df.drop_duplicates(inplace = True)
df.duplicated().sum()


# 5. Check the unique categories of the column 'Category', Is there any invalid category? If yes, drop them.
# 

# In[9]:


df['Category'].unique()


# In[10]:


df=df.drop(df[df['Category']=='1.9'].index)

# category 1.9 is invaild so it is dropped


# 6. Check if there are missing values present in the column Rating, If any? drop them and and create a new
# column as 'Rating_category' by converting ratings to high and low categories(>3.5 is high rest low)
# 

# In[11]:


df['Rating'].isnull().sum()


# In[12]:


df.dropna(inplace = True)


# In[13]:


df[['Rating']]


# In[14]:


df.isnull().sum()


# In[15]:


#create a new column as 'Rating_category' by converting ratings to high and low categories(>3.5 is high rest low)

df['Rating_category']=df['Rating'].apply (lambda a:'high' if a>3.5 else 'low')
df['Rating_category']


# 7. Check the distribution of the newly created column 'Rating_category' and comment on the distribution

# In[16]:


# plot pie chart of the newly created column 

plt.pie(df.groupby(['Rating_category'])['Rating_category'].count(),
        autopct='%1.1f%%',labels=df.groupby(['Rating_category'])['Rating_category'].count().index)
plt.title('Rating_category')
plt.show()


# 8. Convert the column "Reviews'' to numeric data type and check the presence of outliers in the column and
# handle the outliers using a transformation approach.(Hint: Use log transformation)
# 

# In[17]:


df['Reviews']=pd.to_numeric(df['Reviews'],errors='coerce')


# In[18]:


plt.figure(figsize=(5,3))
plt.boxplot(df['Reviews'].dropna())
plt.show()


# In[19]:


df['Reviews']=np.log1p(df['Reviews'])
plt.boxplot(df['Reviews'].dropna())
plt.show()


# 9. The column 'Size' contains alphanumeric values, treat the non numeric data and convert the column into
# suitable data type. (hint: Replace M with 1 million and K with 1 thousand, and drop the entries where
# size='Varies with device')
# 

# In[20]:


df['Size']=df['Size'].str.replace('M','*10**6').str.replace('k','*1000')
df['Size']


# In[21]:


df=df.drop(df[df['Size']=='Varies with device'].index)
df


# In[22]:


df['Size']=df['Size'].map(eval).astype(float)


# In[23]:


df['Size']


# 10. Check the column 'Installs', treat the unwanted characters and convert the column into a suitable data type.
# 

# In[25]:


df['Installs'].unique()


# In[26]:


df['Installs']=df['Installs'].str.replace(',','').astype(str)
# ',' has been removed


# In[27]:


df['Installs'].unique()  


# 11. Check the column 'Price' , remove the unwanted characters and convert the column into a suitable data type.
# 

# In[29]:


df['Price'].unique()


# In[30]:


df["Price"]=df['Price'].str.replace('$',"").astype(float)


# In[32]:


df['Price'].unique()


# 12. Drop the columns which you think redundant for the analysis.(suggestion: drop column 'rating', since we
# created a new feature from it (i.e. rating_category) and the columns 'App', 'Rating' ,'Genres','Last Updated',
# 'Current Ver','Android Ver' columns since which are redundant for our analysis)
# 

# In[33]:


df=df.drop(columns=['Current Ver','App','Rating','Genres','Last Updated','Android Ver','Size'])
df


# In[34]:


num_cols = df.select_dtypes(include=np.number).columns
num_cols


# 13. Encode the categorical columns.
# 

# In[35]:


cat_cols = df.select_dtypes(exclude = np.number).columns
cat_cols


# In[37]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
for i in cat_cols:
    df[i]=le.fit_transform(df[i])


# In[38]:


df


# 14. Segregate the target and independent features (Hint: Use Rating_category as the target)
# 

# In[39]:


target=df[['Rating_category']]
target


# In[40]:


independent_features=df.drop('Rating_category',axis=1)
independent_features


# 15. Split the dataset into train and test.
# 

# In[42]:


x=independent_features
y=target


# In[43]:


from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=1)


# 16. Standardize the data, so that the values are within a particular range.
# 

# In[44]:


from sklearn.preprocessing import StandardScaler
standardization=StandardScaler()
df=standardization.fit_transform(df)
df=pd.DataFrame(df)
df


# In[ ]:




