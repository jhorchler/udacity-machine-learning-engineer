# Overview

## Project Overview

In this project you will apply unsupervised learning techniques on product
spending data collected for customers of a wholesale distributor in Lisbon,
Portugal to identify customer segments hidden in the data. You will first
explore the data by selecting a small subset to sample and determine if any
product categories highly correlate with one another. Afterwards, you will
preprocess the data by scaling each product category and then identifying (and
removing) unwanted outliers. With the good, clean customer spending data, you
will apply PCA transformations to the data and implement clustering algorithms
to segment the transformed customer data. Finally, you will compare the
segmentation found with an additional labeling and consider ways this
information could assist the wholesale distributor with future service changes.

## Project Highlights

This project is designed to give you a hands-on experience with unsupervised
learning and work towards developing conclusions for a potential client on a
real-world dataset. Many companies today collect vast amounts of data on
customers and clientele, and have a strong desire to understand the meaningful
relationships hidden in their customer base. Being equipped with this
information can assist a company with future products and services that best
satisfy the demands or needs of their customers.

Things you will learn by completing this project:

- How to apply preprocessing techniques such as feature scaling and outlier
  detection.
- How to interpret data points that have been scaled, transformed, or reduced
  from PCA.
- How to analyze PCA dimensions and construct a new feature space.
- How to optimally cluster a set of data to find hidden patterns in a dataset.
- How to assess information given by cluster data and use it in a meaningful
  way.

## Data

The customer segments data is included as a selection of 440 data points
collected on data found from clients of a wholesale distributor in Lisbon,
Portugal. More information can be found on the
[UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Wholesale+customers).

Note (m.u.) is shorthand for *monetary units*.

### Features

1) `Fresh`: annual spending (m.u.) on fresh products (Continuous);
2) `Milk`: annual spending (m.u.) on milk products (Continuous);
3) `Grocery`: annual spending (m.u.) on grocery products (Continuous);
4) `Frozen`: annual spending (m.u.) on frozen products (Continuous);
5) `Detergents_Paper`: annual spending (m.u.) on detergents and paper products (Continuous);
6) `Delicatessen`: annual spending (m.u.) on and delicatessen products (Continuous);
7) `Channel`: {Hotel/Restaurant/Cafe - 1, Retail - 2} (Nominal)
8) `Region`: {Lisbon - 1, Oporto - 2, or Other - 3} (Nominal)

## Software Requirements

### Description

A wholesale distributor recently tested a change to their delivery method for
some customers, by moving from a morning delivery service five days a week to a
cheaper evening delivery service three days a week. Initial testing did not
discover any significant unsatisfactory results, so they implemented the cheaper
option for all customers. Almost immediately, the distributor began getting
complaints about the delivery service change and customers were canceling
deliveries - losing the distributor more money than what was being saved. You???ve
been hired by the wholesale distributor to find what types of customers they
have to help them make better, more informed business decisions in the future.
Your task is to use unsupervised learning techniques to see if any similarities
exist between customers, and how to best segment customers into distinct
categories.

### Software and Libraries

This project uses the following software and Python libraries:

- Python
- NumPy
- pandas
- scikit-learn (v0.17)
- matplotlib

You will also need to have software installed to run and execute a Jupyter
Notebook.

If you do not have Python installed yet, it is highly recommended that you
install the Anaconda distribution of Python, which already has the above
packages and more included.

## Starting the project

For this assignment, you can find the `customer_segments` folder containing the
necessary project files on the Machine Learning projects GitHub, under the
`projects` folder. You may download all of the files for projects we'll use in
this Nanodegree program directly from this repo. Please make sure that you use
the most recent version of project files when completing a project!

This project contains three files:

- `customer_segments.ipynb`: This is the main file where you will be performing
  your work on the project.
- `customers.csv`: The project dataset. You'll load this data in the notebook.
- `visuals.py`: This Python script provides supplementary visualizations for the
  project. Do not modify.

In the Terminal or Command Prompt, navigate to the folder containing the project
files, and then use the command `jupyter notebook customer_segments.ipynb` to
open up a browser window or tab to work with your notebook. Alternatively, you
can use the command `jupyter notebook` or `ipython notebook` and navigate to the
notebook file in the browser window that opens. Follow the instructions in the
notebook and answer each question presented to successfully complete the
project. A **README** file has also been provided with the project files which
may contain additional necessary information or instruction for the project.
