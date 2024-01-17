#!/usr/bin/env python
# coding: utf-8

# In[1]:


import warnings
warnings.filterwarnings("ignore")

import pandas as pd

# dataset : https://data.worldbank.org/indicator/AG.LND.FRST.K2?view=chart


# In[2]:


def worldbank_data(path): return pd.read_csv(path, skiprows=4), pd.read_csv(path, skiprows=4).set_index(['Country Name']).T

"""
Read World Bank Forest area (sq. km) data from a CSV file and returns two dataframes.

Returns:
- countries_df: Original dataframe. # Used for Custering
- years_df: Transposed dataframe. # Used for Curve fit

"""


# In[3]:


path = 'API_AG.LND.FRST.K2_DS2_en_csv_v2_6302271.csv'

countries_df, years_df = worldbank_data(path)


# ## Clustering - Forest area (sq. km)

# In[4]:


countries_df.head()


# In[6]:


def extract_year_columns(df, start_year, end_year):
    """
    Extracts specific columns from a DataFrame based on the given range of years.

    Parameters:
    - df: The input DataFrame.
    - start_year: The starting year for column extraction.
    - end_year: The ending year for column extraction.

    Returns:
    pd.DataFrame: A new DataFrame containing the specified columns.
    """
    years = [str(year) for year in range(start_year, end_year + 1)]
    columns_to_extract = ['Country Name', 'Indicator Name'] + years
    return df[columns_to_extract]

start_year = 2001
end_year = 2020

data = extract_year_columns(countries_df, start_year, end_year)


# In[7]:


print(data.isnull().sum())


# In[8]:


data = data.dropna()


# In[9]:


data.head()


# In[33]:


data.describe()


# In[36]:


subset = data[["Country Name", "2020"]].copy()
subset.head()


# In[39]:


subset = subset.assign(Change = lambda x: 100.0 * (data["2020"] - data["2001"]) / data["2001"])


# In[40]:


subset.head()


# In[47]:


subset.isnull().sum()


# In[48]:


subset = subset.dropna()


# In[50]:


subset.describe()


# In[51]:


import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
sns.scatterplot(data=subset, x="2020", y="Change", label="Forest area (sq. km)")
plt.title("Forest area (sq. km) in 2020 vs. Percentage Change (2001 to 2020)")
plt.xlabel("Forest area (sq. km) in 2020")
plt.ylabel("Percentage Change from 2001 to 2020")
plt.show()


# In[52]:


def one_silhouette(xy, n):
    """Calculates silhouette score for n clusters"""
    kmeans = KMeans(n_clusters=n, n_init=20)
    kmeans.fit(xy)
    labels = kmeans.labels_
    score = skmet.silhouette_score(xy, labels)
    return score


# In[53]:


import sklearn.preprocessing as pp
x = subset[["2020", "Change"]].copy()
scaler = pp.RobustScaler()
scaler.fit(x)
x_norm = scaler.transform(x)


# In[57]:


from sklearn.cluster import KMeans
import sklearn.metrics as skmet

silhouette_scores = []

for i in range(2, 12):
    score = one_silhouette(x_norm, i)
    silhouette_scores.append(score)
    print(f"The silhouette score for {i: 3d} is {score: 7.4f}")


# In[60]:


plt.figure(figsize=(8, 5))
sns.lineplot(x=range(2, 12), y=silhouette_scores, marker='o')
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score vs. Number of Clusters')
plt.show()


# In[63]:


kmeans = KMeans(n_clusters=3, n_init=20)
kmeans.fit(x_norm)
labels = kmeans.labels_
centroids = scaler.inverse_transform(kmeans.cluster_centers_)
xkmeans, ykmeans = centroids[:, 0], centroids[:, 1]


# In[65]:


plt.figure(figsize=(8, 5))
plt.scatter(subset["2020"], subset["Change"], 10, labels, marker="o")
plt.scatter(xkmeans, ykmeans, 50, "k", marker="d")
plt.title("Clustes of Forest area (sq. km)")
plt.xlabel("Forest area (sq. km) in 2020")
plt.ylabel("Percentage Change from 2001 to 2020")
plt.show()


# ## Curve Fitting - United Kingdom Forest area (sq. km)

# In[90]:


def error_prop(x, func, parameter, covar):
    """
    Calculates 1 sigma error ranges for number or array. It uses error
    propagation with variances and covariances taken from the covar matrix.
    Derivatives are calculated numerically.
    """
    
    # initiate sigma the same shape as parameter
    var = np.zeros_like(x)   # initialise variance vector
    # Nested loop over all combinations of the parameters
    for i in range(len(parameter)):
        # derivative with respect to the ith parameter
        deriv1 = deriv(x, func, parameter, i)

        for j in range(len(parameter)):
            # derivative with respect to the jth parameter
            deriv2 = deriv(x, func, parameter, j)
            # multiplied with the i-jth covariance
            # variance vector 
            var = var + deriv1 * deriv2 * covar[i, j]

    sigma = np.sqrt(var)
    return sigma


def deriv(x, func, parameter, ip):
    """
    Calculates numerical derivatives from function
    values at parameter +/- delta. Parameter is the vector with parameter
    values. ip is the index of the parameter to derive the derivative.
    """

    # create vector with zeros and insert delta value for the relevant parameter
    scale = 1e-6   # scale factor to calculate the derivative
    delta = np.zeros_like(parameter, dtype=float)
    val = scale * np.abs(parameter[ip])
    delta[ip] = val
    
    diff = 0.5 * (func(x, *parameter+delta) - func(x, *parameter-delta))
    dfdx = diff / val

    return dfdx


def covar_to_corr(covar):
    """ Converts the covariance matrix into a correlation matrix """
    # extract variances from the diagonal and calculate std. dev.
    sigma = np.sqrt(np.diag(covar))
    # construct matrix containing the sigma values
    matrix = np.outer(sigma, sigma)
    # and divide by it
    corr = covar / matrix
    
    return corr


# In[68]:


years_df['United Kingdom']


# In[73]:


uk_data = years_df[['United Kingdom']]
uk_data = uk_data.loc['2001':'2020']
uk_data.reset_index(inplace=True)
uk_data.columns.name = 'Index'
uk_data.rename(columns={'index': 'Years'}, inplace=True)
uk_data.rename(columns={'United Kingdom': 'Forest area (sq. km)'}, inplace=True)


# In[78]:


uk_data.dtypes


# In[79]:


uk_data = uk_data.apply(pd.to_numeric, errors='coerce')


# In[80]:


uk_data.describe()


# In[82]:


plt.figure(figsize=(8, 5))
sns.lineplot(data=uk_data, x='Years', y='Forest area (sq. km)')
plt.xlabel('Years')
plt.ylabel('Forest area (sq. km)')
plt.title('Forest area (sq. km) in United Kingdom between 2001-2020')
plt.show()


# In[89]:


def poly(x, a, b, c, d, e):
    """ Calulates polynominal"""
    x = x - 2001
    f = a + b*x + c*x**2 + d*x**3 + e*x**4
    return f


# In[96]:


import scipy.optimize as opt
import numpy as np

param, covar = opt.curve_fit(poly, uk_data["Years"], uk_data["Forest area (sq. km)"])
sigma = np.sqrt(np.diag(covar))
year = np.arange(2001, 2031)
forecast = poly(year, *param)
sigma = error_prop(year, poly, param, covar)
low = forecast - sigma
up = forecast + sigma
uk_data["fit"] = poly(uk_data["Years"], *param)
plt.figure(figsize=(8, 5))
plt.plot(uk_data["Years"], uk_data["Forest area (sq. km)"], label="Forest area (sq. km)")
plt.plot(year, forecast, label="forecast")
plt.fill_between(year, low, up, color="yellow", alpha=0.7)
plt.title("Forest area (sq. km) Prediction in United Kingdom")
plt.xlabel("Years")
plt.ylabel("Forest area (sq. km)")
plt.legend()
plt.show()


# In[ ]:




