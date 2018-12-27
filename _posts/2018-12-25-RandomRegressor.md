---
bg: "/randf/rf_post.jpg"
layout: post
title:  "RandomForest Regressor with Python"
crawlertitle: "RandomForest"
summary: "RandomForestRegressor with Python"
date:   2018-12-25 02:00:47 +0700
categories: posts
tags: ['Machine Learning']
author: Uzoma Uzosike
---
***Objective***: Predicting the Sale Price of bulldozers sold at Auctions

***Evaluation Criterion***: Root Squared Mean Error

***Data Description***: Access the Data on Kaggle using this [Link](https://www.kaggle.com/c/bluebook-for-bulldozers/data) 

```python
%load_ext autoreload
%autoreload 2
%matplotlib inline
```
***Import Necessary Python Libraries***

```python
import pandas as pd
from fastai.imports import *
from fastai.structured import *

from pandas_summary import DataFrameSummary
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from IPython.display import display

from sklearn import metrics
```
***During Data import, the "saledate" column is parsed as a datetime object, because the column was recorded in a datetime format, the datetime format allows for easy manipulation of datetime object*** 
```python
# Importing the data, and setting "saledate" column as datetime format
df = pd.read_csv("../../data/mydata/Train.csv",low_memory=False,parse_dates=["saledate"])
```
### Exploratory Data Analysis
- ***Create a function **display_all()**  to display all columns avoiding hiding excess columns and ensuring a wholistic view of the data set***
```python
def display_all(df):
    with pd.option_context("display.max_rows", 1000, "display.max_columns", 1000): 
        display(df)
```
- ***A Look into the data using the **display_all()  and  df.head()** functions***.
```python
 display_all(df.head(3))
```
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>SalesID</th>
      <th>SalePrice</th>
      <th>MachineID</th>
      <th>ModelID</th>
      <th>datasource</th>
      <th>auctioneerID</th>
      <th>YearMade</th>
      <th>MachineHoursCurrentMeter</th>
      <th>UsageBand</th>
      <th>saledate</th>
      <th>fiModelDesc</th>
      <th>fiBaseModel</th>
      <th>fiSecondaryDesc</th>
      <th>fiModelSeries</th>
      <th>fiModelDescriptor</th>
      <th>ProductSize</th>
      <th>fiProductClassDesc</th>
      <th>state</th>
      <th>ProductGroup</th>
      <th>ProductGroupDesc</th>
      <th>Drive_System</th>
      <th>Enclosure</th>
      <th>Forks</th>
      <th>Pad_Type</th>
      <th>Ride_Control</th>
      <th>Stick</th>
      <th>Transmission</th>
      <th>Turbocharged</th>
      <th>Blade_Extension</th>
      <th>Blade_Width</th>
      <th>Enclosure_Type</th>
      <th>Engine_Horsepower</th>
      <th>Hydraulics</th>
      <th>Pushblock</th>
      <th>Ripper</th>
      <th>Scarifier</th>
      <th>Tip_Control</th>
      <th>Tire_Size</th>
      <th>Coupler</th>
      <th>Coupler_System</th>
      <th>Grouser_Tracks</th>
      <th>Hydraulics_Flow</th>
      <th>Track_Type</th>
      <th>Undercarriage_Pad_Width</th>
      <th>Stick_Length</th>
      <th>Thumb</th>
      <th>Pattern_Changer</th>
      <th>Grouser_Type</th>
      <th>Backhoe_Mounting</th>
      <th>Blade_Type</th>
      <th>Travel_Controls</th>
      <th>Differential_Type</th>
      <th>Steering_Controls</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1139246</td>
      <td>66000</td>
      <td>999089</td>
      <td>3157</td>
      <td>121</td>
      <td>3.0</td>
      <td>2004</td>
      <td>68.0</td>
      <td>Low</td>
      <td>2006-11-16</td>
      <td>521D</td>
      <td>521</td>
      <td>D</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Wheel Loader - 110.0 to 120.0 Horsepower</td>
      <td>Alabama</td>
      <td>WL</td>
      <td>Wheel Loader</td>
      <td>NaN</td>
      <td>EROPS w AC</td>
      <td>None or Unspecified</td>
      <td>NaN</td>
      <td>None or Unspecified</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2 Valve</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>None or Unspecified</td>
      <td>None or Unspecified</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Standard</td>
      <td>Conventional</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1139248</td>
      <td>57000</td>
      <td>117657</td>
      <td>77</td>
      <td>121</td>
      <td>3.0</td>
      <td>1996</td>
      <td>4640.0</td>
      <td>Low</td>
      <td>2004-03-26</td>
      <td>950FII</td>
      <td>950</td>
      <td>F</td>
      <td>II</td>
      <td>NaN</td>
      <td>Medium</td>
      <td>Wheel Loader - 150.0 to 175.0 Horsepower</td>
      <td>North Carolina</td>
      <td>WL</td>
      <td>Wheel Loader</td>
      <td>NaN</td>
      <td>EROPS w AC</td>
      <td>None or Unspecified</td>
      <td>NaN</td>
      <td>None or Unspecified</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2 Valve</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>23.5</td>
      <td>None or Unspecified</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Standard</td>
      <td>Conventional</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1139249</td>
      <td>10000</td>
      <td>434808</td>
      <td>7009</td>
      <td>121</td>
      <td>3.0</td>
      <td>2001</td>
      <td>2838.0</td>
      <td>High</td>
      <td>2004-02-26</td>
      <td>226</td>
      <td>226</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Skid Steer Loader - 1351.0 to 1601.0 Lb Operat...</td>
      <td>New York</td>
      <td>SSL</td>
      <td>Skid Steer Loaders</td>
      <td>NaN</td>
      <td>OROPS</td>
      <td>None or Unspecified</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Auxiliary</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>None or Unspecified</td>
      <td>None or Unspecified</td>
      <td>None or Unspecified</td>
      <td>Standard</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>

- ***A function **df_summary(df)** that summarizes the data, providing the number of instances, features and data types for each cloumn/feature contained in the data***

```python
def df_summary(df):
    print ("There are {a} columns \nThere are {b} entries in this DataFrame"
    .format(a=df.shape[1],b=df.shape[0]))
    cols= []
    typs= []
    for col,typ in zip(list(df.columns),list(df.dtypes)):
        cols.append(col)
        typs.append(typ)
    summary = pd.DataFrame(data=typs,index=cols,columns=["Data Type"])
    return summary.T

```


```python
display_all(df_summary(df))
```

    There are 53 columns 
    There are 401125 entries in this DataFrame
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>SalesID</th>
      <th>SalePrice</th>
      <th>MachineID</th>
      <th>ModelID</th>
      <th>datasource</th>
      <th>auctioneerID</th>
      <th>YearMade</th>
      <th>MachineHoursCurrentMeter</th>
      <th>UsageBand</th>
      <th>saledate</th>
      <th>fiModelDesc</th>
      <th>fiBaseModel</th>
      <th>fiSecondaryDesc</th>
      <th>fiModelSeries</th>
      <th>fiModelDescriptor</th>
      <th>ProductSize</th>
      <th>fiProductClassDesc</th>
      <th>state</th>
      <th>ProductGroup</th>
      <th>ProductGroupDesc</th>
      <th>Drive_System</th>
      <th>Enclosure</th>
      <th>Forks</th>
      <th>Pad_Type</th>
      <th>Ride_Control</th>
      <th>Stick</th>
      <th>Transmission</th>
      <th>Turbocharged</th>
      <th>Blade_Extension</th>
      <th>Blade_Width</th>
      <th>Enclosure_Type</th>
      <th>Engine_Horsepower</th>
      <th>Hydraulics</th>
      <th>Pushblock</th>
      <th>Ripper</th>
      <th>Scarifier</th>
      <th>Tip_Control</th>
      <th>Tire_Size</th>
      <th>Coupler</th>
      <th>Coupler_System</th>
      <th>Grouser_Tracks</th>
      <th>Hydraulics_Flow</th>
      <th>Track_Type</th>
      <th>Undercarriage_Pad_Width</th>
      <th>Stick_Length</th>
      <th>Thumb</th>
      <th>Pattern_Changer</th>
      <th>Grouser_Type</th>
      <th>Backhoe_Mounting</th>
      <th>Blade_Type</th>
      <th>Travel_Controls</th>
      <th>Differential_Type</th>
      <th>Steering_Controls</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Data Type</th>
      <td>int64</td>
      <td>int64</td>
      <td>int64</td>
      <td>int64</td>
      <td>int64</td>
      <td>float64</td>
      <td>int64</td>
      <td>float64</td>
      <td>object</td>
      <td>datetime64[ns]</td>
      <td>object</td>
      <td>object</td>
      <td>object</td>
      <td>object</td>
      <td>object</td>
      <td>object</td>
      <td>object</td>
      <td>object</td>
      <td>object</td>
      <td>object</td>
      <td>object</td>
      <td>object</td>
      <td>object</td>
      <td>object</td>
      <td>object</td>
      <td>object</td>
      <td>object</td>
      <td>object</td>
      <td>object</td>
      <td>object</td>
      <td>object</td>
      <td>object</td>
      <td>object</td>
      <td>object</td>
      <td>object</td>
      <td>object</td>
      <td>object</td>
      <td>object</td>
      <td>object</td>
      <td>object</td>
      <td>object</td>
      <td>object</td>
      <td>object</td>
      <td>object</td>
      <td>object</td>
      <td>object</td>
      <td>object</td>
      <td>object</td>
      <td>object</td>
      <td>object</td>
      <td>object</td>
      <td>object</td>
      <td>object</td>
    </tr>
  </tbody>
</table>
</div>

- ***Converting the Target variable (SalePrice) into it's Logarithmic Values ensuring compatibility to evaluation requirements RSME***

```python
df["SalePrice"] = np.log(df["SalePrice"])
```
- ***A look at the statistical illustration of the data***
```python
df.describe(include="all")
```
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>SalesID</th>
      <th>SalePrice</th>
      <th>MachineID</th>
      <th>ModelID</th>
      <th>datasource</th>
      <th>auctioneerID</th>
      <th>YearMade</th>
      <th>MachineHoursCurrentMeter</th>
      <th>UsageBand</th>
      <th>saledate</th>
      <th>...</th>
      <th>Undercarriage_Pad_Width</th>
      <th>Stick_Length</th>
      <th>Thumb</th>
      <th>Pattern_Changer</th>
      <th>Grouser_Type</th>
      <th>Backhoe_Mounting</th>
      <th>Blade_Type</th>
      <th>Travel_Controls</th>
      <th>Differential_Type</th>
      <th>Steering_Controls</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>4.011250e+05</td>
      <td>401125.000000</td>
      <td>4.011250e+05</td>
      <td>401125.000000</td>
      <td>401125.000000</td>
      <td>380989.000000</td>
      <td>401125.000000</td>
      <td>1.427650e+05</td>
      <td>69639</td>
      <td>401125</td>
      <td>...</td>
      <td>99872</td>
      <td>99218</td>
      <td>99288</td>
      <td>99218</td>
      <td>99153</td>
      <td>78672</td>
      <td>79833</td>
      <td>79834</td>
      <td>69411</td>
      <td>69369</td>
    </tr>
    <tr>
      <th>unique</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>3</td>
      <td>3919</td>
      <td>...</td>
      <td>19</td>
      <td>29</td>
      <td>3</td>
      <td>3</td>
      <td>3</td>
      <td>2</td>
      <td>10</td>
      <td>7</td>
      <td>4</td>
      <td>5</td>
    </tr>
    <tr>
      <th>top</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Medium</td>
      <td>2009-02-16 00:00:00</td>
      <td>...</td>
      <td>None or Unspecified</td>
      <td>None or Unspecified</td>
      <td>None or Unspecified</td>
      <td>None or Unspecified</td>
      <td>Double</td>
      <td>None or Unspecified</td>
      <td>PAT</td>
      <td>None or Unspecified</td>
      <td>Standard</td>
      <td>Conventional</td>
    </tr>
    <tr>
      <th>freq</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>33985</td>
      <td>1932</td>
      <td>...</td>
      <td>79651</td>
      <td>78820</td>
      <td>83093</td>
      <td>90255</td>
      <td>84653</td>
      <td>78652</td>
      <td>38612</td>
      <td>69923</td>
      <td>68073</td>
      <td>68679</td>
    </tr>
    <tr>
      <th>first</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1989-01-17 00:00:00</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>last</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2011-12-30 00:00:00</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>1.919713e+06</td>
      <td>10.103096</td>
      <td>1.217903e+06</td>
      <td>6889.702980</td>
      <td>134.665810</td>
      <td>6.556040</td>
      <td>1899.156901</td>
      <td>3.457955e+03</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>std</th>
      <td>9.090215e+05</td>
      <td>0.693621</td>
      <td>4.409920e+05</td>
      <td>6221.777842</td>
      <td>8.962237</td>
      <td>16.976779</td>
      <td>291.797469</td>
      <td>2.759026e+04</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.139246e+06</td>
      <td>8.465900</td>
      <td>0.000000e+00</td>
      <td>28.000000</td>
      <td>121.000000</td>
      <td>0.000000</td>
      <td>1000.000000</td>
      <td>0.000000e+00</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>1.418371e+06</td>
      <td>9.581904</td>
      <td>1.088697e+06</td>
      <td>3259.000000</td>
      <td>132.000000</td>
      <td>1.000000</td>
      <td>1985.000000</td>
      <td>0.000000e+00</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>1.639422e+06</td>
      <td>10.085809</td>
      <td>1.279490e+06</td>
      <td>4604.000000</td>
      <td>132.000000</td>
      <td>2.000000</td>
      <td>1995.000000</td>
      <td>0.000000e+00</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>2.242707e+06</td>
      <td>10.596635</td>
      <td>1.468067e+06</td>
      <td>8724.000000</td>
      <td>136.000000</td>
      <td>4.000000</td>
      <td>2000.000000</td>
      <td>3.025000e+03</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>max</th>
      <td>6.333342e+06</td>
      <td>11.863582</td>
      <td>2.486330e+06</td>
      <td>37198.000000</td>
      <td>172.000000</td>
      <td>99.000000</td>
      <td>2013.000000</td>
      <td>2.483300e+06</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>13 rows × 53 columns</p>
</div>

#### Performing operations on the DateTime object column "saledate"
Using **add_datepart** method from the **fastai** Library, We split the **"saledate"** column into seperate colums, specifiy a seperate new column for each element of datetime, like: **Year, Month, Day...**
                      
```python
add_datepart(df,"saledate")
```

- Looking at the data summary once again shows that the number of features increased initially from **53 columns to 65**. 12 elements of date time were created as new columns
```python
display_all(df_summary(df))
```
```python
There are 65 columns 
There are 401125 entries in this DataFrame    
```
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>SalesID</th>
      <th>SalePrice</th>
      <th>MachineID</th>
      <th>ModelID</th>
      <th>datasource</th>
      <th>auctioneerID</th>
      <th>YearMade</th>
      <th>MachineHoursCurrentMeter</th>
      <th>UsageBand</th>
      <th>fiModelDesc</th>
      <th>fiBaseModel</th>
      <th>fiSecondaryDesc</th>
      <th>fiModelSeries</th>
      <th>fiModelDescriptor</th>
      <th>ProductSize</th>
      <th>fiProductClassDesc</th>
      <th>state</th>
      <th>ProductGroup</th>
      <th>ProductGroupDesc</th>
      <th>Drive_System</th>
      <th>Enclosure</th>
      <th>Forks</th>
      <th>Pad_Type</th>
      <th>Ride_Control</th>
      <th>Stick</th>
      <th>Transmission</th>
      <th>Turbocharged</th>
      <th>Blade_Extension</th>
      <th>Blade_Width</th>
      <th>Enclosure_Type</th>
      <th>Engine_Horsepower</th>
      <th>Hydraulics</th>
      <th>Pushblock</th>
      <th>Ripper</th>
      <th>Scarifier</th>
      <th>Tip_Control</th>
      <th>Tire_Size</th>
      <th>Coupler</th>
      <th>Coupler_System</th>
      <th>Grouser_Tracks</th>
      <th>Hydraulics_Flow</th>
      <th>Track_Type</th>
      <th>Undercarriage_Pad_Width</th>
      <th>Stick_Length</th>
      <th>Thumb</th>
      <th>Pattern_Changer</th>
      <th>Grouser_Type</th>
      <th>Backhoe_Mounting</th>
      <th>Blade_Type</th>
      <th>Travel_Controls</th>
      <th>Differential_Type</th>
      <th>Steering_Controls</th>
      <th>saleYear</th>
      <th>saleMonth</th>
      <th>saleWeek</th>
      <th>saleDay</th>
      <th>saleDayofweek</th>
      <th>saleDayofyear</th>
      <th>saleIs_month_end</th>
      <th>saleIs_month_start</th>
      <th>saleIs_quarter_end</th>
      <th>saleIs_quarter_start</th>
      <th>saleIs_year_end</th>
      <th>saleIs_year_start</th>
      <th>saleElapsed</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Data Type</th>
      <td>int64</td>
      <td>float64</td>
      <td>int64</td>
      <td>int64</td>
      <td>int64</td>
      <td>float64</td>
      <td>int64</td>
      <td>float64</td>
      <td>object</td>
      <td>object</td>
      <td>object</td>
      <td>object</td>
      <td>object</td>
      <td>object</td>
      <td>object</td>
      <td>object</td>
      <td>object</td>
      <td>object</td>
      <td>object</td>
      <td>object</td>
      <td>object</td>
      <td>object</td>
      <td>object</td>
      <td>object</td>
      <td>object</td>
      <td>object</td>
      <td>object</td>
      <td>object</td>
      <td>object</td>
      <td>object</td>
      <td>object</td>
      <td>object</td>
      <td>object</td>
      <td>object</td>
      <td>object</td>
      <td>object</td>
      <td>object</td>
      <td>object</td>
      <td>object</td>
      <td>object</td>
      <td>object</td>
      <td>object</td>
      <td>object</td>
      <td>object</td>
      <td>object</td>
      <td>object</td>
      <td>object</td>
      <td>object</td>
      <td>object</td>
      <td>object</td>
      <td>object</td>
      <td>object</td>
      <td>int64</td>
      <td>int64</td>
      <td>int64</td>
      <td>int64</td>
      <td>int64</td>
      <td>int64</td>
      <td>bool</td>
      <td>bool</td>
      <td>bool</td>
      <td>bool</td>
      <td>bool</td>
      <td>bool</td>
      <td>int64</td>
    </tr>
  </tbody>
</table>
</div>

- ***Using the **train_cats()** from fastai, we convert the String Object Columns of the data into Categorical Variables***
```python
train_cats(df)
```
Notice the change in type of the column "UsageBand", from previously "object" into "category"
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>SalesID</th>
      <th>SalePrice</th>
      <th>MachineID</th>
      <th>ModelID</th>
      <th>datasource</th>
      <th>auctioneerID</th>
      <th>YearMade</th>
      <th>MachineHoursCurrentMeter</th>
      <th>UsageBand</th>
      <th>fiModelDesc</th>
      <th>fiBaseModel</th>
      <th>fiSecondaryDesc</th>
      <th>fiModelSeries</th>
      <th>fiModelDescriptor</th>
      <th>ProductSize</th>
      <th>fiProductClassDesc</th>
      <th>state</th>
      <th>ProductGroup</th>
      <th>ProductGroupDesc</th>
      <th>Drive_System</th>
      <th>Enclosure</th>
      <th>Forks</th>
      <th>Pad_Type</th>
      <th>Ride_Control</th>
      <th>Stick</th>
      <th>Transmission</th>
      <th>Turbocharged</th>
      <th>Blade_Extension</th>
      <th>Blade_Width</th>
      <th>Enclosure_Type</th>
      <th>Engine_Horsepower</th>
      <th>Hydraulics</th>
      <th>Pushblock</th>
      <th>Ripper</th>
      <th>Scarifier</th>
      <th>Tip_Control</th>
      <th>Tire_Size</th>
      <th>Coupler</th>
      <th>Coupler_System</th>
      <th>Grouser_Tracks</th>
      <th>Hydraulics_Flow</th>
      <th>Track_Type</th>
      <th>Undercarriage_Pad_Width</th>
      <th>Stick_Length</th>
      <th>Thumb</th>
      <th>Pattern_Changer</th>
      <th>Grouser_Type</th>
      <th>Backhoe_Mounting</th>
      <th>Blade_Type</th>
      <th>Travel_Controls</th>
      <th>Differential_Type</th>
      <th>Steering_Controls</th>
      <th>saleYear</th>
      <th>saleMonth</th>
      <th>saleWeek</th>
      <th>saleDay</th>
      <th>saleDayofweek</th>
      <th>saleDayofyear</th>
      <th>saleIs_month_end</th>
      <th>saleIs_month_start</th>
      <th>saleIs_quarter_end</th>
      <th>saleIs_quarter_start</th>
      <th>saleIs_year_end</th>
      <th>saleIs_year_start</th>
      <th>saleElapsed</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Data Type</th>
      <td>int64</td>
      <td>float64</td>
      <td>int64</td>
      <td>int64</td>
      <td>int64</td>
      <td>float64</td>
      <td>int64</td>
      <td>float64</td>
      <td>category</td>
      <td>category</td>
      <td>category</td>
      <td>category</td>
      <td>category</td>
      <td>category</td>
      <td>category</td>
      <td>category</td>
      <td>category</td>
      <td>category</td>
      <td>category</td>
      <td>category</td>
      <td>category</td>
      <td>category</td>
      <td>category</td>
      <td>category</td>
      <td>category</td>
      <td>category</td>
      <td>category</td>
      <td>category</td>
      <td>category</td>
      <td>category</td>
      <td>category</td>
      <td>category</td>
      <td>category</td>
      <td>category</td>
      <td>category</td>
      <td>category</td>
      <td>category</td>
      <td>category</td>
      <td>category</td>
      <td>category</td>
      <td>category</td>
      <td>category</td>
      <td>category</td>
      <td>category</td>
      <td>category</td>
      <td>category</td>
      <td>category</td>
      <td>category</td>
      <td>category</td>
      <td>category</td>
      <td>category</td>
      <td>category</td>
      <td>int64</td>
      <td>int64</td>
      <td>int64</td>
      <td>int64</td>
      <td>int64</td>
      <td>int64</td>
      <td>bool</td>
      <td>bool</td>
      <td>bool</td>
      <td>bool</td>
      <td>bool</td>
      <td>bool</td>
      <td>int64</td>
    </tr>
  </tbody>
</table>
</div>

#### Handling Categorical Variables
- ***A look into the categories contained in the  "UsageBand" column***
```python
df["UsageBand"].cat.categories
```
It can seen that there are 3 unique values

```python
Index(['High', 'Low', 'Medium'], dtype='object')
```
- ***Re-odering the Categoric Values***
We place reorder the values stating the desired order of values 

```python
df.UsageBand.cat.set_categories(['High', 'Medium', 'Low'], ordered=True, inplace=True)
```
- ***Values are arranged in desired order***
```python
df.UsageBand.cat.categories
```
      Index(['High', 'Medium', 'Low'], dtype='object')


- ***Then we replace the String names of the category values, with coresponding categorical codes (1,2,3)***

```python
df["UsageBand"] = df["UsageBand"].cat.codes
```


```python
df["UsageBand"].unique()
```

      array([ 2,  0,  1, -1], dtype=int64)


**NOTE** Pandas Library uses *-1* to signify missing values in Categorical Variables. This is why we see *-1* shown as a value  

## Handling Missing Values in the Data
1. **missing_col_count()** : Counts and Returns all columns containing Missing Values from the Dataset

2. **report_missing ()** : Uses the missing_col_count() to print result of columns contain missing values

3. **missing_values_summary()** : Gives a wholistic summary of missing values including a dataframe generated to show columns containing missing values, and the corresponding percentage of missing data per column

- **missing_col_count()**
```python
def missing_col_count(df):
    missing = df.isnull().sum()/len(df)*100
    count = []
    percentage = []
    for col in missing.index:
        if missing[col] > 0:
            count.append(col)
        percentage.append(round(missing[col],2)) 
    return (len(count))
```
- **report_missing ()**
```python
def report_missing (df):
    m = missing_col_count(df)
    print("{a} of {b} Columns contain Missing Values in the Dataframe".format(a=m,b=len(df.columns)))
```


```python
report_missing(df)
```
    39 of 65 Columns contain Missing Values in the Dataframe
    
- **missing_values_summary()**

```python
def missing_values_summary(df):
    try:
        # Total missing values
        mis_val = df.isnull().sum()
        # Percentage of missing values
        mis_val_percent = 100 * df.isnull().sum() / len(df)
        
        # Make a table with the results
        mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
        
        # Rename the columns
        mis_val_table_ren_columns = mis_val_table.rename(
        columns = {0 : 'Missing Values', 1 : 'Total Values(%)'})
        
        # Sort the table by percentage of missing descending
        mis_val_table_ren_columns = mis_val_table_ren_columns[
            mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(
        'Total Values(%)', ascending=False).round(1)
        
        # Print some summary information
        print ("Your selected dataframe has {a} columns.\n There are {b} columns that have missing values."
               .format(a=df.shape[1],b=mis_val_table_ren_columns.shape[0]))
        # Return the dataframe with missing information
        return mis_val_table_ren_columns
    except AttributeError:
        print("There are no Missing Values in this Data")

```


```python
missing_values_summary(df)
```

    Your selected dataframe has 65 columns.
    There are 39 columns that have missing values.
    

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Missing Values</th>
      <th>Total Values(%)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Pushblock</th>
      <td>375906</td>
      <td>93.7</td>
    </tr>
    <tr>
      <th>Blade_Extension</th>
      <td>375906</td>
      <td>93.7</td>
    </tr>
    <tr>
      <th>Tip_Control</th>
      <td>375906</td>
      <td>93.7</td>
    </tr>
    <tr>
      <th>Engine_Horsepower</th>
      <td>375906</td>
      <td>93.7</td>
    </tr>
    <tr>
      <th>Enclosure_Type</th>
      <td>375906</td>
      <td>93.7</td>
    </tr>
    <tr>
      <th>Blade_Width</th>
      <td>375906</td>
      <td>93.7</td>
    </tr>
    <tr>
      <th>Scarifier</th>
      <td>375895</td>
      <td>93.7</td>
    </tr>
    <tr>
      <th>Hydraulics_Flow</th>
      <td>357763</td>
      <td>89.2</td>
    </tr>
    <tr>
      <th>Grouser_Tracks</th>
      <td>357763</td>
      <td>89.2</td>
    </tr>
    <tr>
      <th>Coupler_System</th>
      <td>357667</td>
      <td>89.2</td>
    </tr>
    <tr>
      <th>fiModelSeries</th>
      <td>344217</td>
      <td>85.8</td>
    </tr>
    <tr>
      <th>Steering_Controls</th>
      <td>331756</td>
      <td>82.7</td>
    </tr>
    <tr>
      <th>Differential_Type</th>
      <td>331714</td>
      <td>82.7</td>
    </tr>
    <tr>
      <th>fiModelDescriptor</th>
      <td>329206</td>
      <td>82.1</td>
    </tr>
    <tr>
      <th>Backhoe_Mounting</th>
      <td>322453</td>
      <td>80.4</td>
    </tr>
    <tr>
      <th>Stick</th>
      <td>321991</td>
      <td>80.3</td>
    </tr>
    <tr>
      <th>Pad_Type</th>
      <td>321991</td>
      <td>80.3</td>
    </tr>
    <tr>
      <th>Turbocharged</th>
      <td>321991</td>
      <td>80.3</td>
    </tr>
    <tr>
      <th>Blade_Type</th>
      <td>321292</td>
      <td>80.1</td>
    </tr>
    <tr>
      <th>Travel_Controls</th>
      <td>321291</td>
      <td>80.1</td>
    </tr>
    <tr>
      <th>Tire_Size</th>
      <td>306407</td>
      <td>76.4</td>
    </tr>
    <tr>
      <th>Track_Type</th>
      <td>301972</td>
      <td>75.3</td>
    </tr>
    <tr>
      <th>Grouser_Type</th>
      <td>301972</td>
      <td>75.3</td>
    </tr>
    <tr>
      <th>Stick_Length</th>
      <td>301907</td>
      <td>75.3</td>
    </tr>
    <tr>
      <th>Pattern_Changer</th>
      <td>301907</td>
      <td>75.3</td>
    </tr>
    <tr>
      <th>Thumb</th>
      <td>301837</td>
      <td>75.2</td>
    </tr>
    <tr>
      <th>Undercarriage_Pad_Width</th>
      <td>301253</td>
      <td>75.1</td>
    </tr>
    <tr>
      <th>Ripper</th>
      <td>296988</td>
      <td>74.0</td>
    </tr>
    <tr>
      <th>Drive_System</th>
      <td>296764</td>
      <td>74.0</td>
    </tr>
    <tr>
      <th>MachineHoursCurrentMeter</th>
      <td>258360</td>
      <td>64.4</td>
    </tr>
    <tr>
      <th>Ride_Control</th>
      <td>252519</td>
      <td>63.0</td>
    </tr>
    <tr>
      <th>Transmission</th>
      <td>217895</td>
      <td>54.3</td>
    </tr>
    <tr>
      <th>ProductSize</th>
      <td>210775</td>
      <td>52.5</td>
    </tr>
    <tr>
      <th>Forks</th>
      <td>209048</td>
      <td>52.1</td>
    </tr>
    <tr>
      <th>Coupler</th>
      <td>187173</td>
      <td>46.7</td>
    </tr>
    <tr>
      <th>fiSecondaryDesc</th>
      <td>137191</td>
      <td>34.2</td>
    </tr>
    <tr>
      <th>Hydraulics</th>
      <td>80555</td>
      <td>20.1</td>
    </tr>
    <tr>
      <th>auctioneerID</th>
      <td>20136</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>Enclosure</th>
      <td>325</td>
      <td>0.1</td>
    </tr>
  </tbody>
</table>
</div>

### Ploting the Missing Values by the percentage per column
- A dataframe that holds ***Column of Missing Value*** and ***Percentage of Missing Value*** for the corresponding column
```python
def plt_missing(df):
    mis_val_percent = 100 * df.isnull().sum() / len(df)
    mis_val = mis_val_percent[df.isnull().sum() > 0].round(2)
    vals = pd.DataFrame(mis_val)
    vals = vals.reset_index()
    vals = vals.rename(columns={0:'Percentages',"index":'Col_Name'})
    return vals
```
- **Bar plot showing Percentage of Missing Value by for Columns of the Data where Missing values exist**
  - Green: Signifies that below 25% of values are missing in the column
  - Blue: Signifies that below 60% of values are missing in the column
  - Red: Signifies that above 60% of values are missing in the column 

```python
color = lambda x: "green" if x<=25 else("blue" if x<= 60 else "red")
plt.figure(figsize=(16,8))
fig = plt.bar(range(len(plt_missing(df)["Col_Name"])), plt_missing(df)["Percentages"].values,
        color=plt_missing(df)["Percentages"].apply(color))
plt.xticks(range(len(plt_missing(df)["Col_Name"])), plt_missing(df)["Col_Name"], rotation =90)
plt.ylabel('Percentage (%)')
plt.title('Percentage of Missing Values By Column')
plt.legend(fig, ['Green: <=25%','Blue: <= 60%','Red: 60% and Above'], loc = "upper right", title = "Severity")
```
[![png]({{ site.images | relative_url }}/randf1/output_26_1.png)]({{ site.images.randf1 | relative_url }}/randf/output_26_1.png)



#### Tidying Up Data 
  - Saving the Current Dataframe and every change that has been made so far as a feather file
  - We create a folder in the current directory
```python
os.makedirs("temp",exist_ok=True)
```
- We save the file with a user defined name
```python
df.to_feather("temp/data_now")
```

##### At a later time, we simply call up the feather file and continue from previous point
```python
import feather
df_new = feather.read_dataframe("temp/data_now")

```
```python
df_summary(df_new)
```

    There are 65 columns 
    There are 401125 entries in this DataFrame
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>SalesID</th>
      <th>SalePrice</th>
      <th>MachineID</th>
      <th>ModelID</th>
      <th>datasource</th>
      <th>auctioneerID</th>
      <th>YearMade</th>
      <th>MachineHoursCurrentMeter</th>
      <th>UsageBand</th>
      <th>fiModelDesc</th>
      <th>...</th>
      <th>saleDay</th>
      <th>saleDayofweek</th>
      <th>saleDayofyear</th>
      <th>saleIs_month_end</th>
      <th>saleIs_month_start</th>
      <th>saleIs_quarter_end</th>
      <th>saleIs_quarter_start</th>
      <th>saleIs_year_end</th>
      <th>saleIs_year_start</th>
      <th>saleElapsed</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Data Type</th>
      <td>int64</td>
      <td>float64</td>
      <td>int64</td>
      <td>int64</td>
      <td>int64</td>
      <td>float64</td>
      <td>int64</td>
      <td>float64</td>
      <td>int8</td>
      <td>category</td>
      <td>...</td>
      <td>int64</td>
      <td>int64</td>
      <td>int64</td>
      <td>bool</td>
      <td>bool</td>
      <td>bool</td>
      <td>bool</td>
      <td>bool</td>
      <td>bool</td>
      <td>int64</td>
    </tr>
  </tbody>
</table>
<p>1 rows × 65 columns</p>
</div>




```python
df_new.head(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>SalesID</th>
      <th>SalePrice</th>
      <th>MachineID</th>
      <th>ModelID</th>
      <th>datasource</th>
      <th>auctioneerID</th>
      <th>YearMade</th>
      <th>MachineHoursCurrentMeter</th>
      <th>UsageBand</th>
      <th>fiModelDesc</th>
      <th>...</th>
      <th>saleDay</th>
      <th>saleDayofweek</th>
      <th>saleDayofyear</th>
      <th>saleIs_month_end</th>
      <th>saleIs_month_start</th>
      <th>saleIs_quarter_end</th>
      <th>saleIs_quarter_start</th>
      <th>saleIs_year_end</th>
      <th>saleIs_year_start</th>
      <th>saleElapsed</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1139246</td>
      <td>11.097410</td>
      <td>999089</td>
      <td>3157</td>
      <td>121</td>
      <td>3.0</td>
      <td>2004</td>
      <td>68.0</td>
      <td>2</td>
      <td>521D</td>
      <td>...</td>
      <td>16</td>
      <td>3</td>
      <td>320</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>1163635200</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1139248</td>
      <td>10.950807</td>
      <td>117657</td>
      <td>77</td>
      <td>121</td>
      <td>3.0</td>
      <td>1996</td>
      <td>4640.0</td>
      <td>2</td>
      <td>950FII</td>
      <td>...</td>
      <td>26</td>
      <td>4</td>
      <td>86</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>1080259200</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1139249</td>
      <td>9.210340</td>
      <td>434808</td>
      <td>7009</td>
      <td>121</td>
      <td>3.0</td>
      <td>2001</td>
      <td>2838.0</td>
      <td>0</td>
      <td>226</td>
      <td>...</td>
      <td>26</td>
      <td>3</td>
      <td>57</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>1077753600</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1139251</td>
      <td>10.558414</td>
      <td>1026470</td>
      <td>332</td>
      <td>121</td>
      <td>3.0</td>
      <td>2001</td>
      <td>3486.0</td>
      <td>0</td>
      <td>PC120-6E</td>
      <td>...</td>
      <td>19</td>
      <td>3</td>
      <td>139</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>1305763200</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1139253</td>
      <td>9.305651</td>
      <td>1057373</td>
      <td>17311</td>
      <td>121</td>
      <td>3.0</td>
      <td>2007</td>
      <td>722.0</td>
      <td>1</td>
      <td>S175</td>
      <td>...</td>
      <td>23</td>
      <td>3</td>
      <td>204</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>1248307200</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 65 columns</p>
</div>



#### Preprocessing with the **proc_df()** in ***fastai***
- proc_df() is responsible for the following:
  - Replace Categories with Numeric codes
  - Imputation of missing continuous values by filling with the Median Values 
  - Spliting the Target Variable off from the Dataframe into a Separate variable (Y).
  - Adding a new column ***nas*** which signifies columns where missing values were imputed by the median

```python
df_ready, y, na_vals = proc_df(df_new,"SalePrice")
```
```python
### 
display_all(df_summary(df_ready))
```
Notice that the target has been moved off from the data frame as y

    There are 66 columns 
    There are 401125 entries in this DataFrame
    
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>SalesID</th>
      <th>MachineID</th>
      <th>ModelID</th>
      <th>datasource</th>
      <th>auctioneerID</th>
      <th>YearMade</th>
      <th>MachineHoursCurrentMeter</th>
      <th>UsageBand</th>
      <th>fiModelDesc</th>
      <th>fiBaseModel</th>
      <th>fiSecondaryDesc</th>
      <th>fiModelSeries</th>
      <th>fiModelDescriptor</th>
      <th>ProductSize</th>
      <th>fiProductClassDesc</th>
      <th>state</th>
      <th>ProductGroup</th>
      <th>ProductGroupDesc</th>
      <th>Drive_System</th>
      <th>Enclosure</th>
      <th>Forks</th>
      <th>Pad_Type</th>
      <th>Ride_Control</th>
      <th>Stick</th>
      <th>Transmission</th>
      <th>Turbocharged</th>
      <th>Blade_Extension</th>
      <th>Blade_Width</th>
      <th>Enclosure_Type</th>
      <th>Engine_Horsepower</th>
      <th>Hydraulics</th>
      <th>Pushblock</th>
      <th>Ripper</th>
      <th>Scarifier</th>
      <th>Tip_Control</th>
      <th>Tire_Size</th>
      <th>Coupler</th>
      <th>Coupler_System</th>
      <th>Grouser_Tracks</th>
      <th>Hydraulics_Flow</th>
      <th>Track_Type</th>
      <th>Undercarriage_Pad_Width</th>
      <th>Stick_Length</th>
      <th>Thumb</th>
      <th>Pattern_Changer</th>
      <th>Grouser_Type</th>
      <th>Backhoe_Mounting</th>
      <th>Blade_Type</th>
      <th>Travel_Controls</th>
      <th>Differential_Type</th>
      <th>Steering_Controls</th>
      <th>saleYear</th>
      <th>saleMonth</th>
      <th>saleWeek</th>
      <th>saleDay</th>
      <th>saleDayofweek</th>
      <th>saleDayofyear</th>
      <th>saleIs_month_end</th>
      <th>saleIs_month_start</th>
      <th>saleIs_quarter_end</th>
      <th>saleIs_quarter_start</th>
      <th>saleIs_year_end</th>
      <th>saleIs_year_start</th>
      <th>saleElapsed</th>
      <th>auctioneerID_na</th>
      <th>MachineHoursCurrentMeter_na</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Data Type</th>
      <td>int64</td>
      <td>int64</td>
      <td>int64</td>
      <td>int64</td>
      <td>float64</td>
      <td>int64</td>
      <td>float64</td>
      <td>int8</td>
      <td>int16</td>
      <td>int16</td>
      <td>int16</td>
      <td>int8</td>
      <td>int16</td>
      <td>int8</td>
      <td>int8</td>
      <td>int8</td>
      <td>int8</td>
      <td>int8</td>
      <td>int8</td>
      <td>int8</td>
      <td>int8</td>
      <td>int8</td>
      <td>int8</td>
      <td>int8</td>
      <td>int8</td>
      <td>int8</td>
      <td>int8</td>
      <td>int8</td>
      <td>int8</td>
      <td>int8</td>
      <td>int8</td>
      <td>int8</td>
      <td>int8</td>
      <td>int8</td>
      <td>int8</td>
      <td>int8</td>
      <td>int8</td>
      <td>int8</td>
      <td>int8</td>
      <td>int8</td>
      <td>int8</td>
      <td>int8</td>
      <td>int8</td>
      <td>int8</td>
      <td>int8</td>
      <td>int8</td>
      <td>int8</td>
      <td>int8</td>
      <td>int8</td>
      <td>int8</td>
      <td>int8</td>
      <td>int64</td>
      <td>int64</td>
      <td>int64</td>
      <td>int64</td>
      <td>int64</td>
      <td>int64</td>
      <td>bool</td>
      <td>bool</td>
      <td>bool</td>
      <td>bool</td>
      <td>bool</td>
      <td>bool</td>
      <td>int64</td>
      <td>bool</td>
      <td>bool</td>
    </tr>
  </tbody>
</table>
</div>

***Now or Data set is cleaned, rid of missing values, contains numerical codings for categories, we can proceed to model bulding***

## .......Just Before Model Building  
- Creating a Training set and a validation set is an extremely important step in Machine Learning tasks
- The concept of validation set reduces the the risk of a model **overfiting or underfitting** on the training data. Read more about [Underfitting and Overfitting HERE](https://datascience.stackexchange.com/questions/361/when-is-a-model-underfitted)
   [![png]({{ site.images | relative_url }}/randf1/overfit.png)]({{ site.images.randf1 | relative_url }}/randf/overfit.png)
- Validation set enables possibility of estimating how well the model has been trained by estimating **prediction error of the model**
- Validation set enables one determine the need for tunning model properties where necessary



- ***df_splitter()**: Takes a dataframe and splits it into specified size. One set as Trainig set, while the other is used as validation set
```python
def df_splitter(df,size):
    '''
    Takes a df and required size = No. of Required İnstances
    Returns copy of df according to required size
    '''
    return df[size:].copy(),df[:size].copy()
```


```python
### Splitting into Training and Validation Sets
### The Dataframes are splitted maintaining the original sequence of instance arrangement

df_train, df_valid = df_splitter(df_ready, 20000)

y_train, y_valid = df_splitter(y, 20000)
```
```python
### Splitted frames are as follows
df_train.shape, df_valid.shape, y_train.shape, y_valid.shape
```

    ((381125, 66), (20000, 66), (381125,), (20000,))



### Building the Model
- ***First, we build a function to help track the RootMeanSquaredError (RMSE) for our model***
    - ***rmse_track()*** :  Calculates the RootMeanSquaredError
    - ***print_score()*** : 
      - Calculates the RMSE using the rmse_track() for the training set,and validation set. 
      - Shows the ***Mean Squared Error Score*** for training and validation sets.
      - Shows ***oob_score*** if oob_score is set to "True"

```python
def rmse_track(x,y): 
    ''' 
    x represents predicted values form our model
    y represents the original values of our target variable
    '''
    return math.sqrt(((x-y)**2).mean())


def print_score(m):
    '''This function prints the RSME Training set and Validation set
       We can see model performance on validation set, and tunning necessity
    '''
    res = [rmse_track(m.predict(df_ready_train), y_train), rmse_track(m.predict(df_ready_valid), y_valid),
                m.score(df_ready_train, y_train), m.score(df_ready_valid, y_valid)]
    if hasattr(m, 'oob_score_'): res.append(m.oob_score_)
    print(res)
```

#### Building a Base model for Quick View
- Building a base model of only one tree, using random values and untunned parameters

```python
base_mod = RandomForestRegressor(n_jobs=-1,n_estimators=1,bootstrap=False)
```
```python
%time base_mod.fit(df_ready_train,y_train)
```

    Wall time: 11.1 s
    
    RandomForestRegressor(bootstrap=False, criterion='mse', max_depth=None,
               max_features='auto', max_leaf_nodes=None,
               min_impurity_decrease=0.0, min_impurity_split=None,
               min_samples_leaf=1, min_samples_split=2,
               min_weight_fraction_leaf=0.0, n_estimators=1, n_jobs=-1,
               oob_score=False, random_state=None, verbose=0, warm_start=False)

**A look at the RMSE scores on training set and validation set with the print_score()
 
```python
print_score(base_mod)
```
    [1.0983985613061447e-06, 0.36086465048321226, 0.9999999999974865, 0.7367468865904693]
    
#### ......Bagging to the Rescue
Bagging **(BootStrapped Aggregation)**, is a machine learning ensemble method designed to improve the stability and accuracy of machine learning algorithms, It reduces variance and helps to avoid overfitting.<br/> 
***Random Forest*** is a bagging approach because it involves **taking random samples from the data** (bootstrapping) and using the ***insights*** from these samples to build ***numerous uncorrelated models***, The generated models are **Averaged (Regression tasks)** or **Voted (Classification task)** to obtain the optimal model. <br/> 
*Bagging ensures each individual estimator is optimally predictive yet uncorrelated*<br/>

Bagging was proposed by **Leo Breiman in 1994**. Read more [Here](https://en.wikipedia.org/wiki/Bootstrap_aggregating)


A good Model:
  - Predicts the taining set accurately
  - Effective establishes the relationships that exists in Training Set
  - Generalizes well on New Unseen Data

```python
bagged_mod = RandomForestRegressor(n_jobs=-1)
bagged_mod.fit(df_ready_train, y_train)
print_score(bagged_mod)
```

    [0.0905261270376209, 0.2606287462554886, 0.9829267705212436, 0.8626813697498997]
    
***Since bagging generates more than one estimator, and calculates the average to give the best model, unlike our base_model, we can see that the RSME on validation improved significantly***

***The default number of estimators is 10***<br/>
***For better insight, we plot the RSME score to show the impact of estimator(tree) increament showing to illustrate bagging***


```python

# preds hold the prediction result for each estimator
preds = np.stack([tree.predict(df_ready_valid) for tree in bagged_mod.estimators_])

# We take a look at predictions for the first instance, and compare the mean of predictions against the original value
preds[:,0], np.mean(preds[:,0]), y_valid[0]
```
    (array([11.09741, 11.03489, 11.14186, 10.98529, 11.15625, 11.01863, 10.78932, 11.03489, 11.09741, 11.03489]),
     11.039084228167859,
     11.097410021008562)


```python
plt.figure(figsize=(16,8))
plt.ylabel('RSME SCORE')
plt.xlabel('N_ESTIMATORS')
plt.title('Effect of Bagging')
plt.plot([metrics.r2_score(y_valid, np.mean(preds[:i+1], axis=0)) for i in range(10)])
```

[![png]({{ site.images | relative_url }}/randf1/output_51_1.png)]({{ site.images.randf1 | relative_url }}/randf/output_51_1.png)



#### Model Tunning and Hyper-Parameters
- **n_estimators** : Specifies the number of estimators/trees that consitutes the RandomForest
- **oob_score** (True/False) - Allows use of unselected features during individual estimator training as validation data
- **set_rf_samples()** : Sets a number of samples () randomly selected for each estimator
- **min_samples_leaf** reduces overfitting by reducing the depth of tree
- **set_rf_samples()** sets a number of samples to be chosen randomly from the training set for each estimator
- ***max_features*** specifies the number of features to randomly select from at each split

####  ***Comparing results on Tunned Parameters*** 
  - ***n_estimators***

```python

bagged_mod = RandomForestRegressor(n_jobs=-1,n_estimators=20)
bagged_mod.fit(df_ready_train, y_train)
print_score(bagged_mod)
```

    [0.08269300380110428, 0.24898860710162252, 0.9857535934882052, 0.874673248071638]
    


```python
bagged_mod = RandomForestRegressor(n_jobs=-1,n_estimators=40)
bagged_mod.fit(df_ready_train, y_train)
print_score(bagged_mod)
```

    [0.07835619605843071, 0.24623203453723758, 0.9872087058756192, 0.8774328916910471]
    

As was illustrated by the shape of the **"Effect of Bagging"** curve above, after a certain number of n_estimators, increasing the n_estimators will not result any significant model score change

  - **oob_score** - Allows use of unselected features during individual estimator training as validation data

```python
bagged_mod = RandomForestRegressor(n_jobs=-1,n_estimators=40,oob_score=True)
bagged_mod.fit(df_ready_train, y_train)
print_score(bagged_mod)
```
    [0.07828725234185964, 0.24764255861982026, 0.9872312054717943, 0.8760246343515274, 0.9085439037989442]

      Notice the ***oob_score*** is given as ***0.9085439037989442*** 

  - **set_rf_samples()** sets a number of samples to be chosen randomly from the training set for each estimator
Increases the likelyhood of the estimators randomly sampling all instances during training of individual estimator

```python
set_rf_samples(20000)
```

```python
bagged_mod = RandomForestRegressor(n_jobs=-1,n_estimators=40)
bagged_mod.fit(df_ready_train, y_train)
print_score(bagged_mod)
```

    [0.22693450540905882, 0.26767819730540754, 0.8927076228293983, 0.8551525577806699]
    

```python
reset_rf_samples()
```

  - ***min_samples_leaf*** reduces overfitting by reducing the depth of tree 
      - Sample parameters : [1,3,10,25]
  - ***max_features*** specifies the number of features to randomly select from at each split 
      - Sample Parameters: [None,0.5,sqrt,log2]



```python
bagged_mod = RandomForestRegressor(n_jobs=-1,n_estimators=40,min_samples_leaf=3, max_features=0.5,oob_score=True)
bagged_mod.fit(df_ready_train, y_train)
print_score(bagged_mod)
```

    [0.1195504702760178, 0.23154810867437062, 0.9702237244699964, 0.8916154674588154, 0.9115849124625287]

```python
predictions = bagged_mod.predict(df_ready_valid)
```

#### Grid Search
A grid search entails an ***exhausive search for Hyper-parameter values*** by considering ***all possible combinations of parameters*** with the aim of determining the combinations that generates the most optimum results with respect to the prefered evaluation criterion.<br/>Read more on [GridSearchCV  Here](https://scikit-learn.org/stable/modules/grid_search.html)

Note ***estimator.get_params()*** : Gives names and current values for all parameters for a given estimator

```python

from sklearn.model_selection import GridSearchCV
## Parameters to be tuned 
n= list(range(2,42,2))
params = {'n_estimators':n,'min_samples_leaf':[1,3,10,25],'oob_score':['True','False']}

```


```python
df_trn, y_trn, na_val = proc_df(df_new,"SalePrice",subset=50000)
X_train, _ = df_splitter(df_trn, 30000)
y_train, _ = df_splitter(y_trn, 30000)
```


```python
df_vald, y_vald, na_val = proc_df(df_new,"SalePrice",subset=100000)
_, X_val = df_splitter(df_trn, 70000)
_, y_val= df_splitter(y_trn, 70000)
```

```python
## Build an object of the RandomForestRegressor for the grid search
grid_mod = RandomForestRegressor(n_jobs=-1)

## Start the grid search
grid= GridSearchCV(grid_mod,params)

## Fit the grid object with the train data
grid.fit(X_train,y_train)

```

***A Look at the best combination of Parameters after a Grid Search***
```python
grid.best_estimator_
```

    RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=None,
               max_features=0.5, max_leaf_nodes=None,
               min_impurity_decrease=0.0, min_impurity_split=None,
               min_samples_leaf=3, min_samples_split=2,
               min_weight_fraction_leaf=0.0, n_estimators=20, n_jobs=-1,
               oob_score='True', random_state=None, verbose=0,
               warm_start=False)


***Trying out the GridResults on Training and Validation Set***

```python

bagged_mod = RandomForestRegressor(n_jobs=-1,n_estimators=20,min_samples_leaf=3,oob_score=True,min_samples_split=2,
                                   max_features=0.5)
bagged_mod.fit(df_train, y_train)
print_score(bagged_mod)

```
    [0.12208999839611034, 0.23661771975582993, 0.9689452546148812, 0.8868174791256095, 0.8851880113187941]
    

```python
predictions = bagged_mod.predict(df_ready_valid)
```
#### Visual Comparism Between Original Values and Predicted Results from Model
- A distribution plot showing original values versus predicted values
  - Red line represents original Values
  - Blue line represents predicted values
```python
plt.figure(figsize=(16,8))
plt.title('Original Values Versus Predicted Values')
ax1 = sns.distplot(y_valid,hist=False,color="r",label="Original Target Values")
sns.distplot(predictions,hist=False,color="b",label="Predicted Values",ax=ax1)
```
[![png]({{ site.images | relative_url }}/randf1/output_76_2.png)]({{ site.images.randf1 | relative_url }}/randf/output_76_2.png)
