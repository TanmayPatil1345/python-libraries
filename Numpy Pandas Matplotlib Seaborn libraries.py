#!/usr/bin/env python
# coding: utf-8

# # NUMPY 

# In[2]:


import numpy as np


# In[3]:


arr=np.array([1,2,3])


# In[4]:


li=[1,2,3]


# In[5]:


type(arr)


# In[6]:


print("=== NumPy Array Creation ===")
a = np.array([1, 2, 3])
print("1. np.array:", a)

b = np.zeros((2, 3))
print("\n2. np.zeros:\n", b)

c = np.ones((3, 2))
print("\n3. np.ones:\n", c)

d = np.full((2, 2), 7)
print("\n4. np.full:\n", d)

e = np.eye(3)
print("\n5. np.eye:\n", e)

f = np.arange(0, 10, 2)
print("\n6. np.arange:", f)

g = np.linspace(0, 1, 10)
print("\n7. np.linspace:", g)

h = np.random.rand(2, 2)
print("\n8. np.random.rand:\n", h)

i = np.random.randn(3, 3)
print("\n9. np.random.randn:\n", i)

j = np.random.randint(1, 10, (2, 2))
print("\n10. np.random.randint:\n", j)

print("\n=== Shape & Dimension ===")
print("11. Shape of a:", b.shape)
print("12. Size of a:", b.size)
print("13. Dimensions of a:", b.ndim)

k = a.reshape(3, 1)
print("\n14. Reshape a to (3,1):\n", k)

l = np.ravel(k)
print("\n15. Ravel (flatten) k:", l)

m = np.transpose(c)
print("\n16. Transpose of c:\n", m)

n = np.expand_dims(a, axis=0)
print("\n17. Expand dims (axis=0):", n, "Shape:", n.shape)

print("\n=== Mathematical Operations ===")
print("18. Sum of a:", np.sum(a))
print("19. Mean of a:", np.mean(a))
print("20. Standard deviation of a:", np.std(a))

print("\n=== Indexing & Slicing ===")
print("21. First element of a:", a[0])
print("22. Slice a[1:3]:", a[1:3])
print("23. Last element (a[-1]):", a[-1])

print("\n=== Logical Operations ===")
mask = a > 1
print("24. Boolean mask (a > 1):", mask)

filtered = a[mask]
print("25. Filtered elements (a > 1):", filtered)

all_gt0 = np.any(a > 1)
print("26. Are all elements > 0?:", all_gt0)

print("\n=== Aggregation & Utility Functions ===")
print("27. Max of a:", np.max(a))
print("28. Min of a:", np.min(a))
print("29. Argmax (index of max in a):", np.argmax(a))
print("30. Unique elements in a:", np.unique(a))


# # PANDAS

# In[8]:


import pandas as pd
import numpy as np

print("=== 1. Creating DataFrames and Series ===")
data = {'Name': ['Alice', 'Bob', 'Charlie', 'David', 'Eva'],
        'Age': [24, 27, 22, 32, 29],
        'Score': [85.5, 90.0, np.nan, 88.5, 95.0]}

df = pd.DataFrame(data)
print("1. DataFrame:\n", df)

s = pd.Series([10, 20, 30])
print("\n2. Series:\n", s)


# In[9]:


type(df)


# In[10]:


type(s)


# In[11]:


df.to_excel('my data.xlsx',index=False)


# In[12]:


new_df=pd.read_excel('my data.xlsx')


# In[13]:


new_df


# In[14]:


print("\n=== 2. Basic Information ===")
print("3. df.head():\n", df.head()) #top 5 rows 
print("4. df.tail():\n", df.tail()) #bottom 5 rows
print("5. df.shape:", df.shape) #(rows,columns)
print("6. df.columns:", df.columns) #column names
print("7. df.index:", df.index) #total rows
print("8. df.dtypes:\n", df.dtypes) #column data types
print("9. df.info():"); df.info() #overall info


# In[15]:


print("\n=== 3. Descriptive Statistics ===")
print("10. df.describe():\n", df.describe())
print("11. df['Age'].mean():", df['Age'].mean())
print("12. df['Age'].median():", df['Age'].median())
print("13. df['Age'].mode():", df['Age'].mode().tolist())
print("14. df['Score'].isnull():\n", df['Score'].isnull())
print("14. df['Score'].isnull():\n", df['Score'].isnull().sum())


# In[16]:


df.head(2)


# In[ ]:





# In[ ]:





# In[17]:


print("\n=== 4. Handling Missing Values ===")
print("15. Fill NaN:\n", df['Score'].fillna(0))
print("16. Drop rows with NaN:\n", df.dropna())

print("\n=== 5. Data Selection & Filtering ===")
print("17. df['Name']:\n", df['Name'])
print("18. df[['Name', 'Age']]:\n", df[['Name', 'Age']])
print("19. df.loc[0]:\n", df.loc[0])
print("20. df.iloc[1]:\n", df.iloc[1])
print("21. df[df['Age'] > 25]:\n", df[df['Age'] > 25])

print("\n=== 6. Modifying Data ===")
df['Passed'] = df['Score'] > 85
print("22. Add 'Passed' column:\n", df)


# In[18]:


df['Passed']=['Yes','No','No','No','No']


# In[19]:


fruits=pd.DataFrame({'kg':[3,4,5],'units':[20,40,50]},index=['apple','mango','banana'])


# In[20]:


df.drop(2,axis=0,inplace=True)


# In[21]:


df.reset_index(drop=True)


# In[22]:


df.drop('Passed',axis=1)


# In[23]:


df.drop([0,1])


# In[24]:


df.drop(['Score','Age'],axis=1)


# In[26]:


df.at[2, 'Score'] = 78
print("23. Modify single value:\n", df)

print("\n=== 7. Sorting & Grouping ===")
print("24. Sorted by Age:\n", df.sort_values(by='Age'))
print("25. Group by Passed:\n", df.groupby('Passed')['Score'].mean())

print("\n=== 8. File I/O (Commented out) ===")
# df.to_csv('output.csv', index=False)  # 26. Save to CSV
# df2 = pd.read_csv('output.csv')      # 27. Read from CSV

print("\n=== 9. Advanced Indexing & Operations ===")
print("26. Unique Ages:", df['Age'].unique())
print("27. Value counts (Age):\n", df['Age'].value_counts())
print("28. Apply function (double Age):\n", df['Age'].apply(lambda x: x * 2))
print("29. Rename columns:\n", df.rename(columns={'Score': 'Marks'}))
print("30. Drop column 'Passed':\n", df.drop(columns=['Passed']))


# # MATPLOTLIB & SEABORN

# In[ ]:





# In[58]:


import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

# Sample Data
np.random.seed(42)
df = pd.DataFrame({
    'Category': np.random.choice(['A', 'B', 'C'], size=100),
    'Value1': np.random.normal(loc=50, scale=10, size=100),
    'Value2': np.random.randint(20, 80, size=100),
    'Score': np.random.rand(100)
})

# Set seaborn theme
sns.set(style="whitegrid")

# 1. Line Plot
plt.figure(figsize=(6, 4))
plt.plot(df['Value1'][:10], label='Line Plot')
plt.title("1. Line Plot")
plt.xlabel("Index")
plt.ylabel("Value1")
plt.legend()
plt.show()

# 2. Scatter Plot
plt.figure(figsize=(6, 4))
plt.scatter(df['Value1'], df['Value2'], c=df['Score'], cmap='viridis')
plt.title("2. Scatter Plot")
plt.xlabel("Value1")
plt.ylabel("Value2")
plt.colorbar(label='Score')
plt.show()

# 3. Bar Plot
plt.figure(figsize=(6, 4))
sns.barplot(x='Category', y='Value1', data=df)
plt.title("3. Bar Plot")
plt.show()

# 4. Count Plot
plt.figure(figsize=(6, 4))
sns.countplot(x='Category', data=df)
plt.title("4. Count Plot")
plt.show()

# 5. Box Plot
plt.figure(figsize=(6, 4))
sns.boxplot(x='Category', y='Value1', data=df)
plt.title("5. Box Plot")
plt.show()

# 6. Violin Plot
plt.figure(figsize=(6, 4))
sns.violinplot(x='Category', y='Value1', data=df)
plt.title("6. Violin Plot")
plt.show()

# 7. Histogram
plt.figure(figsize=(6, 4))
plt.hist(df['Value1'], bins=15, color='skyblue', edgecolor='black')
plt.title("7. Histogram")
plt.xlabel("Value1")
plt.ylabel("Frequency")
plt.show()

# 8. KDE Plot
plt.figure(figsize=(6, 4))
sns.kdeplot(df['Value1'], fill=True)
plt.title("8. KDE Plot")
plt.show()

# 9. Heatmap
plt.figure(figsize=(6, 4))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title("9. Heatmap")
plt.show()

# 10. Pairplot
print("10. Pairplot (shows multiple plots together)")
sns.pairplot(df[['Value1', 'Value2', 'Score']])
plt.show()


# In[ ]:





# In[ ]:




