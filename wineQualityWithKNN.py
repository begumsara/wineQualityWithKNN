## -----------------------------------------------------------------------------------------------------------------------------------
## Title: wineQualityWithKNN class
## Author: Begüm Şara Ünal
## Description: This class estimates the wine quality using the k-nearest neighbor (KNN) algorithm and 
## recommends wine to the user from the data set using the KNN algorithm based on the values entered by the user. 
## The class uses the red wine quality data from kaggle. (Kaggle link is below.) Also, there are graphs for various purposes in this class.
## The link of the dataset taken from Kaggle: https://www.kaggle.com/datasets/uciml/red-wine-quality-cortez-et-al-2009?resource=download
## ------------------------------------------------------------------------------------------------------------------------------------

#import libraries to use 
import pandas as pd #to read data
import seaborn as sns #to show charts etc.
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import numpy as np

##---------------------------------------------------------------------------------------------------
## Summary: Loads dataset from csv document. Then, it is assigned to a DataFrame with the name wine.
##---------------------------------------------------------------------------------------------------
wine = pd.read_csv('winequality-red.csv')

##-------------------------------------------------
## Summary: Print the initial rows and check data
##-------------------------------------------------
print(wine.head())

##----------------------------------------------------
## Summary: Print information about the data columns
##---------------------------------------------------
print(wine.info())


##-------------------------------------------------
## Summary: Print and check the size of data
##-------------------------------------------------
print(wine.shape)


##-------------------------------------------------------------------------------------------------
## Summary: Creates and displays a graph where the 'quality' of all data is different in color.
##-------------------------------------------------------------------------------------------------
#sns.pairplot(wine, hue='quality')
#plt.show()

##-------------------------------------------------------------------------------------------------
## Summary: #Seperates indepentent variables (inputs) and target variables (outputs). 
## The x variable is created by leaving the "quality" column from the DataFrame with the arguments. 
## The variable y is set as the "quality" column.
##-------------------------------------------------------------------------------------------------
x = wine.drop("quality", axis =1)
y = wine["quality"]

##---------------------------------------------------------------------------------------------------
## Summary: Using the train_test_split() function, the dataset is split into training and test sets
##---------------------------------------------------------------------------------------------------
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

## ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
## test_size --> The test_size parameter determines how much of the data set will be reserved as the test set. 
## In this case, 20% of the data set is reserved as the test set. In general, the size of the test set can be changed, 
## depending on the size of the data set.
## ----------------------------------------------------------------------------------------------------------------------
## random_state --> The random_state parameter is a seed value used during random splitting of the dataset. 
## This way, the random split of the dataset remains the same on each run. 
## So random_state=42 is a seed value chosen to get the same random split on each run.
## ----------------------------------------------------------------------------------------------------------------------
## The reasons for determining these parameters are to ensure the repeatability of the model 
## and to create an initial set of tests. Thus, you can obtain consistent results when evaluating 
## the performance of the model. The random_state value can be arbitrarily changed, but the advantage of using 
## a specific value is that the same random split can be repeated.
## /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

##--------------------------------------------------------------------------------------------------------------------------
## Summary: This part performs the data scaling. Scaling features ensures that each feature is at the same scale. 
## Most machine learning algorithms want features to be at the same scale. 
## Therefore, scaling features is one of the data processing steps.
##-------------------------------------------------------------------------------------------------------------------------

## /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
## `StandardScaler` class --> We scale features using the `StandardScaler` class. `StandardScaler` is a commonly used class
## for scaling features. The scaling operation converts the mean of each feature to 0 and its standard deviation to 1.
## /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
scaler = StandardScaler()
#The training process calculates the mean and standard deviation of each feature.
x_train = scaler.fit_transform(x_train)  #We train (fit) the 'StandardScaler' using the `X_train` dataset (This operation returns the scaled version of the properties.)
                                         
x_test = scaler.transform(x_test)        #We scale both the training and test datasets using the `transform` method.
                                         #After this process, the `X_train` and `X_test` datasets are ready for use with their features at scale. 
                                         
#*** This helps the model work better without being affected by the different scales of the features and allows the algorithm to observe all features with equal weight.***


##---------------------------------------------------
## Summary: To use k-nn import necessary libraries
##---------------------------------------------------
from sklearn.neighbors import KNeighborsClassifier 

##---------------------------------------------------------------------------------------------------
## Summary: Creates model with using KNeighborsClassifier()
##---------------------------------------------------------------------------------------------------
knn = KNeighborsClassifier(n_neighbors=1) #n_neighbours value are founded by a try
## //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
## n_neighbors --> The n_neighbors parameter determines the number of neighbors used in the K-Nearest Neighbor (KNN) algorithm. 
## KNN uses labels of nearest neighbors to predict  a new data point. The n_neighbors parameter determines this number of neighbors.
## ----------------------------------------------------------------------------------------------------------------------
## ----------------------------------------------------------------------------------------------------------------------
## In this case, it is set to n_neighbors=5. This means that the labels of the 5 nearest neighbors will be used to predict a new data point. 
## The n_neighbors value can be changed depending on the size of the dataset, complexity and other factors. In general, a small value of 
## n_neighbors makes the model simpler, while a large value considers more neighbors and makes the model more complex.
## /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


##---------------------------------------------------------------------------------------------------
## Summary: Trains the model with using fit() method
##---------------------------------------------------------------------------------------------------
knn.fit(x_train,y_train)

##---------------------------------------------------------------------------------------------------
## Summary: The KNN model makes predictions on test data using the predict() function.
##---------------------------------------------------------------------------------------------------
y_pred = knn.predict(x_test)

##---------------------------------------------------------------------------------------------------
## Summary: The success of the model is calculated using the accuracy_score() function and 
## printed to the screen with print().
##---------------------------------------------------------------------------------------------------
accuracy = accuracy_score(y_test,y_pred)
print("Model Accuracy With Using K-Nearest Neighbours (KNN) :",accuracy)
"""
##---------------------------------------------------------------------------------------------------------------
## Summary: A graph is drawn to see how the success of the KNN model changes by changing the n_neighbors values. 
## For each n_neighbors value, the model is trained and the success score is calculated and plotted on the graph.
###-------------------------------------------------------------------------------------------------------------
import matplotlib.pyplot as plt #to create a graph import this

n_values = range(1,21) #among 1-21 numbers n_neighbours values

##---------------------------------------------------------------------------------------------------------------
## Summary: Create an array to store different accuracy changes due to n_values 
###-------------------------------------------------------------------------------------------------------------
accuracy_scores = []

##---------------------------------------------------------------------------------------------------
## Summary: To build, train and calculate accuracy the KNN model for each n_neighbors value
##---------------------------------------------------------------------------------------------------
for n in n_values:
    knn = KNeighborsClassifier(n_neighbors=n)
    knn.fit(x_train,y_train)
    y_pred = knn.predict(x_test)
    accuracy = accuracy_score(y_test,y_pred)
    accuracy_scores.append(accuracy)

##---------------------------------------------------------------------------------------------------
## Summary: Draws a graph which has k values and accuracy
##---------------------------------------------------------------------------------------------------
plt.plot(n_values,accuracy_scores,marker ='o')
plt.xlabel('K Values')
plt.ylabel('Accuracy')
plt.title('Accuracy changes with different k values')
plt.grid(True)
plt.show()


import numpy as np
##---------------------------------------------------------------------------------------------------
## Summary: By taking input from the user, using the KNN model, the index of the closest data 
## in the data set is found and the suggested data is printed.
##---------------------------------------------------------------------------------------------------
fixed_acidity = float(input("Fixed Acidity: "))
volatile_acidity = float(input("Volatile Acidity:"))
citric_acid = float(input("Citric Acid :"))
residual_sugar = int(float(input("Residual Sugar")))
chlorides = float(input("Chlorides :" ))
free_sulfur_dioxide = float(input("Free Sulphur Dioxide: "))
total_sulfur_dioxide = float(input("Total Sulfur Dioxide: "))
density = float(input("Density: "))
pH = float(input("PH Value: "))
sulphates = float(input("Sulphates: "))
alcohol = float(input("Alcohol: "))

##---------------------------------------------------------------------------------------------------
## Summary: Creates an array with user inputs
##---------------------------------------------------------------------------------------------------
user_input = np.array([[fixed_acidity, volatile_acidity, citric_acid, residual_sugar,
                        chlorides, free_sulfur_dioxide, total_sulfur_dioxide,
                        density, pH, sulphates, alcohol]])


##---------------------------------------------------------------------------------------------------
## Summary: Scale features
##---------------------------------------------------------------------------------------------------
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)
scaled_input = scaler.transform(user_input)

##---------------------------------------------------------------------------------------------------
## Summary: Find nearest neighbors
##---------------------------------------------------------------------------------------------------
distances , indices = knn.kneighbors(scaled_input)

##---------------------------------------------------------------------------------------------------
## Summary: Get index of suggested data
##---------------------------------------------------------------------------------------------------
recommended_index = indices[0][0]

##---------------------------------------------------------------------------------------------------
## Summary: Print the suggested data
##---------------------------------------------------------------------------------------------------
recommended_data = wine.iloc[recommended_index]
print("Suggested Data on K-Nearest Neighbours (KNN) : ")
print(recommended_data)

## ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
## iloc --> iloc is an indexing method used in Pandas library to access data in dataframes (DataFrame). 
## The abbreviation "iloc" stands for "integer location".
## ----------------------------------------------------------------------------------------------------------------------
## ----------------------------------------------------------------------------------------------------------------------
## The statement data.iloc[recommended_index] selects the row with the index recommended_index in the data dataframe. 
## Returns a Pandas Series representing this row containing all columns starting with the first column. 
## That is, recommended_data is a Series object that represents a particular row in the dataframe.
## ----------------------------------------------------------------------------------------------------------------------
## ----------------------------------------------------------------------------------------------------------------------
## In this case, the recommended_data variable represents a Series object containing the nearest neighbor's data 
## estimated by the KNN algorithm.
## ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
"""

##---------------------------------------------------------------------------------------------------
## Summary: The 'cosine_distances' function calculates the cosine distance between two vectors. 
## Calculates the cosine distance using the dot product of two vectors and the norms of the vectors.
##---------------------------------------------------------------------------------------------------
def cosine_similarity(vector1,vector2):
    dot_product = np.dot(vector1,vector2)

    norm1 = np.linalg.norm(vector1)
    norm2 = np.linalg.norm(vector2)
    norm_product = norm1 * norm2

    cosine_distance = dot_product / norm_product

    return cosine_distance
##---------------------------------------------------------------------------------------------------
## Formula of Cosine Similarity: cos(@) = A*B/||A||*||B||
##---------------------------------------------------------------------------------------------------


##---------------------------------------------------------------------------------------------------
## Summary: The 'minkowski_distance' function calculates the Minkowski distance between two vectors. 
## The Minkowski distance generalizes the Euclidean distance between two vectors and takes the 
## value of p as a parameter.
##---------------------------------------------------------------------------------------------------
def minkowski_distance (v1,v2,p): #v1 and v2 is vectors and function takes these vectors
    v1_v2 = np.abs(v1-v2) #calculate substraction of v1 and v2
    powerP = np.power(v1_v2,p) #calculate power p of v1_v2
    sum = np.sum(powerP)#calculate absoule sum
    distance_measure = np.power(sum,1/p) #calculate power of 1/p
    return distance_measure #return the calculation result
##---------------------------------------------------------------------------------------------------
## Formula of Minkowski Distance: D(x_i, x_j) = (abs^D_d=1|x_id-x_jd|'p)'(1/p)
##---------------------------------------------------------------------------------------------------


##--------------------------------------------------------------------------------------------------------
## Summary: The 'knn_cosine_distance' function finds the nearest neighbors of the data points according to 
## the cosine distance using the KNN algorithm. Calculates the cosine distance for each data point and 
## adds them to a list. Finally, it sorts by similarity values and selects the K nearest neighbors.
##--------------------------------------------------------------------------------------------------------
def knn_cosine_distance(data,query,k):
    num_samples = len(data) #calculates the length of data
    distances = []

    for i in range(num_samples):
        distance = cosine_similarity(data[i][:11],query[:11])
        distances.append((distance,i))
    distances.sort(reverse=True)#sort by similarity
    neighbors = distances[:k]#choose nearest neighbor
    return neighbors

##------------------------------------------------------------------------------------------------------------
## Summary: The 'knn_minkowski_distance' function finds the nearest neighbors of the data points according to 
## the Minkowski distance using the KNN algorithm. Calculates the Minkowski distance for each data point and 
# adds them to a list. Finally, it sorts by similarity values and selects the K nearest neighbors.
##-----------------------------------------------------------------------------------------------------------
def knn_minkowski_distance(data,query,k,p):
    num_samples = len(data)
    distances = []

    for i in range(num_samples):
        distance = minkowski_distance(data[i][:11],query[:11],p)
        distances.append((distance,i))
    distances.sort(reverse=True)#sort by distance
    neighbors = distances[:k]#choose nearest neighbor
    return neighbors

##---------------------------------------------------------------------------------------------------
## Summary: Converts the 'wine' dataset to a numpy array.
##---------------------------------------------------------------------------------------------------
wine = np.array(wine[1:],dtype=float) # Skip header line, convert data to float

##---------------------------------------------------------------------------------------------------
## Summary: The 'query' variable represents the query vector. This vector is used to 
## measure similarity of data points.
##---------------------------------------------------------------------------------------------------
query = np.array([7.4, 0.7, 0.0, 1.9, 0.076, 11.0, 34.0, 0.9978, 3.51, 0.56, 9.4])

##---------------------------------------------------------------------------------------------------
## Summary: The k_values variable is defined as a list of K values, with values from 1-20.
##---------------------------------------------------------------------------------------------------
k_values = range(1,21) 

##---------------------------------------------------------------------------------------------------
## Summary: Empty lists named 'cosine', 'minkowski', 'euclidean', and 'manhattan' are defined. 
## These lists are used to store the distance values for each K value.
##---------------------------------------------------------------------------------------------------
cosine = []
minkowski = []
euclidean = []
manhattan = []

##---------------------------------------------------------------------------------------------------
## Summary: It records the distance values by applying the KNN algorithm and distance metrics 
## for each K value.
##---------------------------------------------------------------------------------------------------
for k in k_values:
    # Apply the KNN algorithm
    #Calculate cosine similarity
    cosineSimilarity = knn_cosine_distance(wine,query,k)

    #Calculate minkowski distance
    minkowskiDistance = knn_minkowski_distance(wine,query,k,p=3)

    #Calculate euclidean distance
    euclideanDistance = knn_minkowski_distance(wine,query,k,p=2) #In euclidean distance p value is equal to 2 

    #Calculate manhattan distance
    manhattanDistance = knn_minkowski_distance(wine,query,k,p=1) #In manhattan distance p value is equal to 1

    #Record the distance values
    cosine.append([distance for distance, _ in cosineSimilarity])
    minkowski.append([distance for distance, _ in minkowskiDistance])
    euclidean.append([distance for distance, _ in euclideanDistance])
    manhattan.append([distance for distance, _ in manhattanDistance])

##---------------------------------------------------------------------------------------------------
## Summary: Create and plot a graph for each distance metric using lists of distance values.
##---------------------------------------------------------------------------------------------------
#For cosine similarity
plt.figure(figsize=(10,6)) 

for i,k in enumerate(k_values):
    plt.plot(cosine[i],label=f"K={k}")

plt.xlabel('Index')
plt.ylabel('Cosine Similarity')
plt.legend()
plt.title('Cosine Similarity vs. Index for Different K Values')
plt.show()

#For minkowski distance
plt.figure(figsize=(10,6)) 

for i,k in enumerate(k_values):
    plt.plot(minkowski[i],label=f"K={k}")

plt.xlabel('Index')
plt.ylabel('Minkowski Distance')
plt.legend()
plt.title('Minkowski Distance vs. Index for Different K Values')
plt.show()

#For euclidean distance
plt.figure(figsize=(10,6)) 

for i,k in enumerate(k_values):
    plt.plot(euclidean[i],label=f"K={k}")

plt.xlabel('Index')
plt.ylabel('Euclidean Distance')
plt.legend()
plt.title('Euclidean Distance vs. Index for Different K Values')
plt.show()

#For manhattan distance
plt.figure(figsize=(10,6)) 

for i,k in enumerate(k_values):
    plt.plot(manhattan[i],label=f"K={k}")

plt.xlabel('Index')
plt.ylabel('Manhattan Distance')
plt.legend()
plt.title('Manhattan Distance vs. Index for Different K Values')
plt.show()

#print the distance results with final k value
print(f"{k} Neighbors:")
for distance, index in cosineSimilarity:
    print(f"Cosine Distance: {distance}, Data Index: {index}")
print()


for distance, index in minkowskiDistance:
    print(f"Minkowski Distance: {distance}, Data Index: {index}")
print()

for distance, index in euclideanDistance:
    print(f"Euclidean Distance: {distance}, Data Index: {index}")
print()

for distance, index in manhattanDistance:
    print(f"Manhattan Distance: {distance}, Data Index: {index}")

