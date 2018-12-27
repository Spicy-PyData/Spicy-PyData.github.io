---
bg: "/randf1/webmap.jpg"
layout: post
title:  "WebMap with Python and Folium"
crawlertitle: "Python,Folium,Webmap"
summary: "WebMap with Python and Folium"
date:   2018-11-15 02:00:47 +0700
categories: posts
tags: ['python']
author: Uzoma Uzosike
---
***Objective*** : WebMap built with ***Folium Library in Python*** showing Locations of ***Volcanoes*** in the United States 

```python
import pandas as pd
import folium
```

#### Creating the Map object for the Base Map
- **Note that an initial list containing Longitude and Latitude is passed, as a base location**
```python
mymap = folium.Map(location=[36.79,34.53],zoom_start=6,tiles="Mapbox Bright")
```

#### Creating a Feature Group
- Serves as a bucket to hold a group of features to be applied on the map
- Allows for adding multiple features to the map, by calling the **.add_child()** method off the feature group
- Can be toggled on or off
We assing a name **"My Map"** as a reference for the feature group
```python
fg = folium.FeatureGroup(name="My Map")
```

#### Adding Markers
The Marker specifies a particular point of interest on the Map
Markers can be added on a feature group by passing ***folium.Marker()*** as a parameter of the ***.add_child()***
Marker require the following parameters:
  - **Location** : A list containing values for ***Longitude and Latitude coordinates***
  - **popup** : A popup message with description of the Marker
  - **icon** : An icon specifying the point of interest
  - **color** : A color for the marker

```python
fg.add_child(folium.Marker(location=(36.79,34.53),popup="This is Your Location",icon=folium.Icon(color="red")))
mymap.add_child(fg)
mymap.save("Map1.html")
```
#### Multiple Markers
We intend to create markers for multiple points on the map showing locations of **Volcanoes** in the USA
Location of each volcano is given in a **text file**

Using *Pandas* we import the file and take a look as a pandas dataframe

```python
volc = pd.read_csv("Volcanoes.txt")

volc.head()
```
<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>VOLCANX020</th>
      <th>NUMBER</th>
      <th>NAME</th>
      <th>LOCATION</th>
      <th>STATUS</th>
      <th>ELEV</th>
      <th>TYPE</th>
      <th>TIMEFRAME</th>
      <th>LAT</th>
      <th>LON</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>509.0</td>
      <td>1201-01=</td>
      <td>Baker</td>
      <td>US-Washington</td>
      <td>Historical</td>
      <td>3285.0</td>
      <td>Stratovolcanoes</td>
      <td>D3</td>
      <td>48.776798</td>
      <td>-121.810997</td>
    </tr>
    <tr>
      <th>1</th>
      <td>511.0</td>
      <td>1201-02-</td>
      <td>Glacier Peak</td>
      <td>US-Washington</td>
      <td>Tephrochronology</td>
      <td>3213.0</td>
      <td>Stratovolcano</td>
      <td>D4</td>
      <td>48.111801</td>
      <td>-121.111000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>513.0</td>
      <td>1201-03-</td>
      <td>Rainier</td>
      <td>US-Washington</td>
      <td>Dendrochronology</td>
      <td>4392.0</td>
      <td>Stratovolcano</td>
      <td>D3</td>
      <td>46.869801</td>
      <td>-121.751000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>515.0</td>
      <td>1201-05-</td>
      <td>St. Helens</td>
      <td>US-Washington</td>
      <td>Historical</td>
      <td>2549.0</td>
      <td>Stratovolcano</td>
      <td>D1</td>
      <td>46.199799</td>
      <td>-122.181000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>516.0</td>
      <td>1201-04-</td>
      <td>Adams</td>
      <td>US-Washington</td>
      <td>Tephrochronology</td>
      <td>3742.0</td>
      <td>Stratovolcano</td>
      <td>D6</td>
      <td>46.205799</td>
      <td>-121.490997</td>
    </tr>
  </tbody>
</table>
</div>




```python
uz_map = folium.Map(location=[50.79,-123.53], zoom_start=6, tiles="Mapbox Bright")
```


```python
col = lambda a: "green" if a < 1000 else("orange" if 1000 <= a < 3000  else "red") 

fgv= folium.FeatureGroup(name="Volcanoes Group")

for i in volc.index:
    fgv.add_child(folium.CircleMarker(location=(volc["LAT"][i],volc["LON"][i]), radius = 6, fill_opacity= 0.7, color= "grey",
                                 popup="Name: "+volc["NAME"][i]+", Status: "+volc["STATUS"][i]+", Type: "+volc["TYPE"][i]+", Elevation : "+str(volc["ELEV"][i]),
                                 fill_color= col(volc["ELEV"][i]) ))
    
fgp= folium.FeatureGroup(name="Population Color")
fgp.add_child(folium.GeoJson(data=open("world.json","r", encoding= "utf-8-sig").read(),
                              style_function = lambda x : {'fillColor':'green' if x['properties']['POP2005'] < 10000000 
                                                           else ('pink' if 10000000 <= x['properties']['POP2005'] < 20000000 else 'yellow')}))
                   
uz_map.add_child(fgv)
uz_map.add_child(fgp)

uz_map.add_child(folium.LayerControl())

uz_map.save("My_Map.html")
```

[View Generated Map HERE](/postdata/My_Map.html){:target="_blank"}
