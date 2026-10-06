# ============================================================================== #
# Exploratory Data Analysis & Feature Engineering - Google Play Store Dataset   #
# Extracted Code Cells from: 3.0-EDA+And+FE+Google+Playstore (2).ipynb           #
# ============================================================================== #

# %% [Cell 1]
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")

%matplotlib inline

# %% [Cell 2]
df=pd.read_csv('https://raw.githubusercontent.com/krishnaik06/playstore-Dataset/main/googleplaystore.csv')
df.head()

# %% [Cell 3]
df.shape

# %% [Cell 4]
df.info()

# %% [Cell 5]
##summary of the dataset
df.describe()

# %% [Cell 6]
##Missing Values
df.isnull().sum()

# %% [Cell 7]
df.head(2)

# %% [Cell 8]
df['Reviews'].unique()

# %% [Cell 9]
df['Reviews'].astype(int)

# %% [Cell 10]
df['Reviews'].str.isnumeric().sum()

# %% [Cell 11]
df[~df['Reviews'].str.isnumeric()]

# %% [Cell 12]
df_copy=df.copy()

# %% [Cell 13]
df_copy=df_copy.drop(df_copy.index[10472])

# %% [Cell 14]
df_copy[~df_copy['Reviews'].str.isnumeric()]

# %% [Cell 15]
## Convert Review Datatype to int
df_copy['Reviews']=df_copy['Reviews'].astype(int)

# %% [Cell 16]
df_copy.info()

# %% [Cell 17]
df_copy['Size'].unique()

# %% [Cell 18]
19000K==19M

# %% [Cell 19]
df_copy['Size'].isnull().sum()

# %% [Cell 20]
df_copy['Size']=df_copy['Size'].str.replace('M','000')
df_copy['Size']=df_copy['Size'].str.replace('k','')
df_copy['Size']=df_copy['Size'].replace('Varies with device',np.nan)
df_copy['Size']=df_copy['Size'].astype(float)

# %% [Cell 21]
df_copy['Size']

# %% [Cell 22]
df_copy['Installs'].unique()

# %% [Cell 23]
df_copy['Price'].unique()

# %% [Cell 24]
chars_to_remove=['+',',','$']
cols_to_clean=['Installs','Price']
for item in chars_to_remove:
    for cols in cols_to_clean:
        df_copy[cols]=df_copy[cols].str.replace(item,'')

# %% [Cell 25]
df_copy['Price'].unique()

# %% [Cell 26]
df_copy['Installs'].unique()

# %% [Cell 27]
df_copy['Installs']=df_copy['Installs'].astype('int')
df_copy['Price']=df_copy['Price'].astype('float')

# %% [Cell 28]
df_copy.info()

# %% [Cell 29]
## Handlling Last update feature
df_copy['Last Updated'].unique()

# %% [Cell 30]
df_copy['Last Updated']=pd.to_datetime(df_copy['Last Updated'])
df_copy['Day']=df_copy['Last Updated'].dt.day
df_copy['Month']=df_copy['Last Updated'].dt.month
df_copy['Year']=df_copy['Last Updated'].dt.year

# %% [Cell 31]
df_copy.info()

# %% [Cell 32]
df_copy.head()

# %% [Cell 33]
df_copy.to_csv('data/google_cleaned.csv')

# %% [Cell 34]
df_copy.head()

# %% [Cell 35]
df_copy[df_copy.duplicated('App')].shape

# %% [Cell 36]
df_copy=df_copy.drop_duplicates(subset=['App'],keep='first')

# %% [Cell 37]
df_copy.shape

# %% [Cell 38]
numeric_features = [feature for feature in df_copy.columns if df_copy[feature].dtype != 'O']
categorical_features = [feature for feature in df_copy.columns if df_copy[feature].dtype == 'O']

# print columns
print('We have {} numerical features : {}'.format(len(numeric_features), numeric_features))
print('\nWe have {} categorical features : {}'.format(len(categorical_features), categorical_features))

# %% [Cell 39]
## Proportion of count data on categorical columns
for col in categorical_features:
    print(df[col].value_counts(normalize=True)*100)
    print('---------------------------')

# %% [Cell 40]
## Proportion of count data on numerical columns
plt.figure(figsize=(15, 15))
plt.suptitle('Univariate Analysis of Numerical Features', fontsize=20, fontweight='bold', alpha=0.8, y=1.)

for i in range(0, len(numeric_features)):
    plt.subplot(5, 3, i+1)
    sns.kdeplot(x=df_copy[numeric_features[i]],shade=True, color='r')
    plt.xlabel(numeric_features[i])
    plt.tight_layout()

# %% [Cell 41]
# categorical columns
plt.figure(figsize=(20, 15))
plt.suptitle('Univariate Analysis of Categorical Features', fontsize=20, fontweight='bold', alpha=0.8, y=1.)
category = [ 'Type', 'Content Rating']
for i in range(0, len(category)):
    plt.subplot(2, 2, i+1)
    sns.countplot(x=df[category[i]],palette="Set2")
    plt.xlabel(category[i])
    plt.xticks(rotation=45)
    plt.tight_layout()

# %% [Cell 42]
df_copy.head(2)

# %% [Cell 43]
df_copy['Category'].value_counts().plot.pie(y=df_copy['Category'],figsize=(15,16),autopct='%1.1f')

# %% [Cell 44]
## Top 10 App Categories
category = pd.DataFrame(df_copy['Category'].value_counts())        #Dataframe of apps on the basis of category
category.rename(columns = {'Category':'Count'},inplace=True)

# %% [Cell 45]
category

# %% [Cell 46]
## top 10 app
plt.figure(figsize=(15,6))
sns.barplot(x=category.index[:10], y ='Count',data = category[:10],palette='hls')
plt.title('Top 10 App categories')
plt.xticks(rotation=90)
plt.show()

# %% [Cell 47]
df_cat_installs = df_copy.groupby(['Category'])['Installs'].sum().sort_values(ascending = False).reset_index()
df_cat_installs.Installs = df_cat_installs.Installs/1000000000# converting into billions
df2 = df_cat_installs.head(10)
plt.figure(figsize = (14,10))
sns.set_context("talk")
sns.set_style("darkgrid")

ax = sns.barplot(x = 'Installs' , y = 'Category' , data = df2 )
ax.set_xlabel('No. of Installations in Billions')
ax.set_ylabel('')
ax.set_title("Most Popular Categories in Play Store", size = 20)

# %% [Cell 48]
dfa = df_copy.groupby(['Category' ,'App'])['Installs'].sum().reset_index()
dfa = dfa.sort_values('Installs', ascending = False)
apps = ['GAME', 'COMMUNICATION', 'PRODUCTIVITY', 'SOCIAL' ]
sns.set_context("poster")
sns.set_style("darkgrid")

plt.figure(figsize=(40,30))

for i,app in enumerate(apps):
    df2 = dfa[dfa.Category == app]
    df3 = df2.head(5)
    plt.subplot(4,2,i+1)
    sns.barplot(data= df3,x= 'Installs' ,y='App' )
    plt.xlabel('Installation in Millions')
    plt.ylabel('')
    plt.title(app,size = 20)
    
plt.tight_layout()
plt.subplots_adjust(hspace= .3)
plt.show()

# %% [Cell 49]
rating = df_copy.groupby(['Category','Installs', 'App'])['Rating'].sum().sort_values(ascending = False).reset_index()

toprating_apps = rating[rating.Rating == 5.0]
print("Number of 5 rated apps",toprating_apps.shape[0])
toprating_apps.head(1)

# %% [Cell 50]
df_copy.head()

