# NISTTreeDetector

An important goal for our team is to assess the carbon content in trees. With that goal in mind, we have created models to detect trees in the wild and measure their diameters at brest height (1.3 m). This document describes the requirements, installation and how to run our code to detect trees and measure their diameter at breast height. As input, we have a cloud of points.


# Requirements for Installation

Run the following commands to make sure you have the required pieces of software

* conda install jupyter 
* conda config --add channels conda-forge
* pip install opencv-contrib-python
* conda install laspy
* conda install -c conda-forge pdal python-pdal gdal
* conda install -c sirokujira python-pcl
* pip install pclpy
* conda install matplotlib
* conda install pandas
* conda install -c anaconda scipy


Fill TLS benchmarking Point clouds can be obtained here
https://laserscanning.fi/results-available-for-international-benchmarking-of-terrestrial-laser-scanning-methods/

Link to Downsampled NIST and TLS benchmarking point clouds
https://drive.google.com/drive/folders/15aW3Npr9lOdxGrswWrsN9wN0g2Q9pBGo?usp=sharing

