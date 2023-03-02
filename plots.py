# -*- coding: utf-8 -*-
"""
Created on Thu Dec  8 21:24:29 2022

@author: rjara
"""

#%%

from matplotlib import pyplot as plt

#%%
xvalue=(1,2,3,4,5,6,7,8,9)
yvalue=(1,4,3,7,4,2,9,0,3)



# LABEL LEGENDS
plt.xlabel("Letter")
plt.ylabel("frequency")
plt.title("Titulo", fontsize=20, color= "Green")

plt.plot(xvalue, yvalue, label= "recta 1", color= "red", linewidth= 3, linestyle="--")
plt.plot(yvalue, xvalue, label= "recta 2", linestyle=":", marker = 'x' )
plt.legend()

plt.text(5,2, "BOTTOM")

plt.style.use("fivethirtyeight")
plt.show() # una vez q le diste todos los comandos

#%% SCATTERPLOT (puntos)

plt.scatter(xvalue, yvalue, marker="s",alpha= 0.1) #alpha hace borroso por intensidad

#%% BARCHART

plt.bar(xvalue, yvalue, yerr=error)
plt.ylabel("frequency")

#%% Horizontal barchart

plt.barh(y, width, kwargs)


#%% STACKED BAR CHART


plt.bar(xvalue, yvalue, bottom =yvalue1)


#%% HISTOGRAM


plt.hist(dataset, bins= 40, range = (xmin, xmax))

#Normalizar:
    

plt.hist(dataset, density= True)
plt.hist(dataset, density= True)

#%% Logaritmo
plt.xscale("log")

#%% yticks (para q arranque y de 0)

plt.yticks([0,2,4,6,8,10], [0,"2B", "4B", "6B", "8B","10B"])

#%% Lineas

plt.grid(True) # grilla
 

#%% PLOTS WITH DATA

# Get the total number of avocados sold of each size
nb_sold_by_size = avocados.groupby('size')['nb_sold'].sum()

# Create a bar plot of the number of avocados sold by size
nb_sold_by_size.plot(kind='bar')

# Show the plot
plt.show()

# Get the total number of avocados sold on each date
nb_sold_by_date = avocados.groupby('date')['nb_sold'].sum()

# Create a line plot of the number of avocados sold by date
nb_sold_by_date.plot()

# Show the plot
plt.show()

# Scatter plot of avg_price vs. nb_sold with title
avocados.plot(x='nb_sold',y= 'avg_price', title= "Number of avocados sold vs. average price", kind= 'scatter')

plt.show()


# Modify bins to 20
avocados[avocados["type"] == "conventional"]["avg_price"].hist(bins=20,alpha=0.5)

# Modify bins to 20
avocados[avocados["type"] == "organic"]["avg_price"].hist(bins=20,alpha=0.5)

# Add a legend
plt.legend(["conventional", "organic"])

# Show the plot
plt.show()


#%% SEABORN

import seaborn as sns

# Change this scatter plot to have percent literate on the y-axis
sns.scatterplot(x=gdp, y=percent_literate)

# Create count plot with region on the y-axis
sns.countplot(y=region)
sns.countplot(x=region)

# Create a count plot with "Spiders" on the x-axis
sns.countplot(x='Spiders', data=df)

# Change the legend order in the scatter plot
sns.scatterplot(x="absences", y="G3", 
                data=student_data, 
                hue="location", hue_order=['Rural', 'Urban'])

# Create a dictionary mapping subgroup values to colors
palette_colors = {'Rural': "green", 'Urban': "blue"}

# Create a count plot of school with location subgroups
sns.countplot(x='school', data=student_data, hue='location', palette= palette_colors)

#%% REPLOT

# Change this scatter plot to arrange the plots in rows instead of columns
sns.relplot(x="absences", y="G3", 
            data=student_data,
            kind="scatter", 
            col="study_time") #row o col

# Show plot
plt.show()

# Adjust further to add subplots based on family support
sns.relplot(x="G1", y="G3", 
            data=student_data,
            kind="scatter", row= 'famsup', row_order= ['yes','no'], 
            col="schoolsup",
            col_order=["yes", "no"])

# Create scatter plot of horsepower vs. mpg
sns.relplot(x="horsepower", y="mpg", 
            data=mpg, kind="scatter", 
            size="cylinders", hue='cylinders')

# Create a scatter plot of acceleration vs. mpg

sns.relplot(x="acceleration", y="mpg", 
            data=mpg, kind="scatter", 
            style="origin", hue='origin')


# Create line plot

sns.relplot(x='model_year', y='mpg', data=mpg, kind='line')

# Make the shaded area show the standard deviation
sns.relplot(x="model_year", y="mpg",
            data=mpg, kind="line", ci='sd')

# Add markers and make each line have the same style
sns.relplot(x="model_year", y="horsepower", 
            data=mpg, kind="line", 
            ci=None, style="origin", 
            hue="origin", markers=True, dashes=False)

#%% CATEGORICAL PLOTS

# Separate into column subplots based on age category
sns.catplot(x="Internet usage", data=survey_data, kind="count", col='Age Category')


# Create a bar plot of interest in math, separated by gender

sns.catplot(x='Gender', y='Interested in Math', data= survey_data, kind='bar' )


# List of categories from lowest to highest
category_order = ["<2 hours", 
                  "2 to 5 hours", 
                  "5 to 10 hours", 
                  ">10 hours"]

# Turn off the confidence intervals
sns.catplot(x="study_time", y="G3",
            data=student_data,
            kind="bar",
            order=category_order, ci=None)



# Create a box plot with subgroups and omit the outliers
sns.catplot(x='internet', y='G3', data=student_data, kind='box', hue='location', sym='')


# Set the whiskers at the min and max values
sns.catplot(x="romantic", y="G3",
            data=student_data,
            kind="box",
            whis=[0, 100])


# Remove the lines joining the points
sns.catplot(x="famrel", y="absences",
			data=student_data,
            kind="point",
            capsize=0.2, join=False)

# Import median function from numpy
from numpy import median

# Plot the median number of absences instead of the mean
sns.catplot(x="romantic", y="absences",
			data=student_data,
            kind="point",
            hue="school",
            ci=None, estimator= median)


#%%CUSTOMIZE

# Change the color palette to "RdBu"
sns.set_style("whitegrid")
sns.set_palette("RdBu")

# Create a count plot of survey responses
category_order = ["Never", "Rarely", "Sometimes", 
                  "Often", "Always"]

sns.catplot(x="Parents Advice", 
            data=survey_data, 
            kind="count", 
            order=category_order)

sns.set_context("poster")
sns.set_context("talk")
sns.set_context("notebook")
sns.set_context('paper')

# Set a custom color palette
sns.set_palette(["#39A7D0","#36ADA4"])

# Identify plot type
type_of_g = type(g)

# Add a title "Car Weight vs. Horsepower"
g.fig.suptitle('Car Weight vs. Horsepower', y=1.05)

# Add a title "Average MPG Over Time"
g.set_title("Average MPG Over Time")

# Add x-axis and y-axis labels
g.set(xlabel='Car Model Year', ylabel= 'Average MPG')

# Rotate x-tick labels
plt.xticks(rotation=90)


#%% Advanced Scatter

# Make a scatter plot
plt.plot(age, weight, 'o', alpha=0.1)


# Select the first 1000 respondents
brfss = brfss[:1000]

# Add jittering to age
age = brfss['AGE'] + np.random.normal(0,2.5, size=len(brfss))
# Extract weight
weight = brfss['WTKG3']

# Make a scatter plot
plt.plot(age, weight, 'o', alpha=0.2, markersize=5)


#%% VIOLIN AND BOXPLOTS

# Drop rows with missing data
data = brfss.dropna(subset=['_HTMG10', 'WTKG3'])

# Make a box plot
sns.boxplot(x='_HTMG10',y= 'WTKG3', data= data, whis=10)

# Plot the y-axis on a log scale
plt.yscale('log')

# Remove unneeded lines and label axes
sns.despine(left=True, bottom=True)
plt.xlabel('Height in cm')
plt.ylabel('Weight in kg')
plt.show()


# Drop rows with missing data
data = brfss.dropna(subset=['INCOME2', 'HTM4'])

# Make a violin plot
sns.violinplot(y='HTM4', x='INCOME2', data=data, inner=None)


# Remove unneeded lines and label axes
sns.despine(left=True, bottom=True)
plt.xlabel('Income level')
plt.ylabel('Height in cm')
plt.show()



















