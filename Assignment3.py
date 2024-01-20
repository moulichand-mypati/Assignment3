# coding: utf-8


import numpy as np
import pandas as pd
import seaborn as sns
import scipy.optimize as opt
import matplotlib.pyplot as plt
import sklearn.preprocessing as pp
from sklearn.cluster import KMeans
import sklearn.metrics as skmet
import errors
import warnings
warnings.filterwarnings("ignore")


def worldbank_data(path) :
    """
    Read World Bank Forest area (sq. km) data from a CSV file
    and returns two dataframes.

    Returns:
    - countries_df: Original dataframe. # Used for Custering
    - years_df: Transposed dataframe. # Used for Curve fit

    """
    return pd.read_csv(path, skiprows=4), \
        pd.read_csv(path, skiprows=4).set_index(['Country Name']).T


def extract_year_columns(df, start_year, end_year):
    """
    Extracts specific columns from a DataFrame based on the
    given range of years.

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


def one_silhouette(xy, n):
    """Calculates silhouette score for n clusters"""
    kmeans = KMeans(n_clusters=n, n_init=20)
    kmeans.fit(xy)
    labels = kmeans.labels_
    score = skmet.silhouette_score(xy, labels)
    return score


def poly(x, a, b, c, d, e):
    """ Calulates polynominal"""
    x = x - 2001
    f = a + b*x + c*x**2 + d*x**3 + e*x**4
    return f


path = 'API_AG.LND.FRST.K2_DS2_en_csv_v2_6302271.csv'
countries_df, years_df = worldbank_data(path)
countries_df.head()

start_year = 2001
end_year = 2020
data = extract_year_columns(countries_df, start_year, end_year)
print(data.isnull().sum())
data = data.dropna()
print(data.head())
print(data.describe())

subset = data[["Country Name", "2020"]].copy()
print(subset.head())
subset = subset.assign(
    Change = lambda x: 100.0 * (data["2020"] - data["2001"]) / data["2001"])
subset.head()
print(subset.isnull().sum())
subset = subset.dropna()
subset.describe()

plt.figure(figsize=(10, 6))
sns.scatterplot(
    data=subset, x="2020", y="Change", label="Forest area (sq. km)")
plt.title("Forest area (sq. km) in 2020 vs. Percentage Change (2001 to 2020)")
plt.xlabel("Forest area (sq. km) in 2020")
plt.ylabel("Percentage Change from 2001 to 2020")
plt.show()

x = subset[["2020", "Change"]].copy()
scaler = pp.RobustScaler()
scaler.fit(x)
x_norm = scaler.transform(x)

silhouette_scores = []
for i in range(2, 12):
    score = one_silhouette(x_norm, i)
    silhouette_scores.append(score)
    print(f"The silhouette score for {i: 3d} is {score: 7.4f}")


plt.figure(figsize=(8, 5))
sns.lineplot(x=range(2, 12), y=silhouette_scores, marker='o')
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score vs. Number of Clusters')
plt.show()

kmeans = KMeans(n_clusters=3, n_init=20)
kmeans.fit(x_norm)
labels = kmeans.labels_
centroids = scaler.inverse_transform(kmeans.cluster_centers_)
xkmeans, ykmeans = centroids[:, 0], centroids[:, 1]

plt.figure(figsize=(8, 5))
plt.scatter(subset["2020"], subset["Change"], 10, labels, marker="o")
plt.scatter(xkmeans, ykmeans, 50, "k", marker="d")
plt.title("Clustes of Forest area (sq. km)")
plt.xlabel("Forest area (sq. km) in 2020")
plt.ylabel("Percentage Change from 2001 to 2020")
plt.show()

uk_data = years_df[['United Kingdom']]
uk_data = uk_data.loc['2001':'2020']
uk_data.reset_index(inplace=True)
uk_data.columns.name = 'Index'
uk_data.rename(columns={'index': 'Years'}, inplace=True)
uk_data.rename(
    columns={'United Kingdom': 'Forest area (sq. km)'}, inplace=True)
print(uk_data.dtypes)
uk_data = uk_data.apply(pd.to_numeric, errors='coerce')
print(uk_data.describe())

plt.figure(figsize=(8, 5))
sns.lineplot(data=uk_data, x='Years', y='Forest area (sq. km)')
plt.xlabel('Years')
plt.ylabel('Forest area (sq. km)')
plt.title('Forest area (sq. km) in United Kingdom between 2001-2020')
plt.show()

param, covar = opt.curve_fit(
    poly, uk_data["Years"], uk_data["Forest area (sq. km)"])
sigma = np.sqrt(np.diag(covar))
year = np.arange(2001, 2031)
forecast = poly(year, *param)
sigma = errors.error_prop(year, poly, param, covar)
low = forecast - sigma
up = forecast + sigma
uk_data["fit"] = poly(uk_data["Years"], *param)
plt.figure(figsize=(8, 5))
plt.plot(
    uk_data["Years"], uk_data["Forest area (sq. km)"],
    label="Forest area (sq. km)")
plt.plot(year, forecast, label="forecast")
plt.fill_between(year, low, up, color="yellow", alpha=0.7)
plt.title("Forest area (sq. km) Prediction in United Kingdom")
plt.xlabel("Years")
plt.ylabel("Forest area (sq. km)")
plt.legend()
plt.show()
