# Titanic
## Author: Nishant Nayar

# RMS Titanic
**RMS Titanic** was a British passenger liner, operated by the White Star Line, which sank in the North Atlantic Ocean on 15 April 1912 after striking an iceberg during her maiden voyage from Southampton, UK, to New York City. Of the estimated 2,224 passengers and crew aboard, more than 1,500 died, which made the sinking possibly one of the deadliest for a single ship up to that time.[a] It remains to this day the deadliest peacetime sinking of a superliner or cruise ship. The disaster drew much public attention, provided foundational material for the disaster film genre, and has inspired many artistic works. 

*source*: [Wikipedia](https://en.wikipedia.org/wiki/Titanic)

## Background

Titanic example is the first project that every data enthusiast starts with. There are many articles and topics that talk about the underlying statistics and data science concepts. This is an effort to make this project understandable for all those who are not versed in statistics and concepts of data science. The emphasis here is to make the machine learning models explainable to all.

# Data
Let's first explore the data set that hase been shared with us.

## Age and Sex
Certain age groups have better chances of survival. In this section we will explore the age groups that have better chances of survival over others

### Females
Females between the ages of 14 - 35 have better chances of survival.
![Females](https://github.com/nishantnayar/Titanic/blob/main/img/FemaleSurvival.png?raw=true)

### Males
Infants and males between ages 25-35 have better chances of survival.
![Males](https://github.com/nishantnayar/Titanic/blob/main/img/MaleSurvival.png?raw=true)

### Port of Embarkation

Males and Females who embarked from port of Cherbourg have better chances of survival
![Port](https://github.com/nishantnayar/Titanic/blob/main/img/Embarkation.png?raw=true)


### Ticket Class
Passengers with the first class ticket have better chances of survival

![Ticket Class](https://github.com/nishantnayar/Titanic/blob/main/img/Sex-Ticket.png?raw=true)

### Relatives
The dataset already contains number of siblings / spouses as well as number of parents / children aboard the Titanic. We will add the two data points to identify the number of relatives on board the titanic. And identify the impact of number of relatives have on the chances of survival.

Additionally we will create a _new attribute "Not Alone"_ based on the the number of relatives calculated

![Relatives](https://github.com/nishantnayar/Titanic/blob/main/img/Relatives.png?raw=true)

## Data Cleaning

A quick glance at the Test and Train data given in tables below we observe only three attributes have missing values

* Cabin
* Age
* Port of Embarkation

**Test Data**

![TestData](https://github.com/nishantnayar/Titanic/blob/main/img/train_data.png?raw=true)

**Train Data**

![TrainData](https://github.com/nishantnayar/Titanic/blob/main/img/test_data.png?raw=true)

### Calculating missing values

* **Cabin**: Cabin data starts with an alphabet denoting the deck number where the cabins are located. We will extract the first alphabet and create a new attribute called as **deck**. The alphabet will be assiged a numeric value for ease of model building. The mising values will be assigned a value of 0. Cabin column will be subsequently dropped.
* **Age**: We will randomly assign a non-zero age value between the mean and standard deviation of the existing age values.
* **Embarked**: Since there are only two values missing we will assume them they also embarked from Southampton 'S' as most of the passengers embarked from that port.

### Dropping columns

* **Cabin**: Since we have extracted the deck the value, we can drop this column.
* **Ticket**: There are 681 unique values and difficult to convert into categories.

### Converting columns

For machine learning models to work properly we have to convert each of the attributes into numercial values.

* **Fare**: The attribute is a float value with decimal points and will be converted to integer that will round it off to a whole number.
* **Name**: There are common titles and some rare titles, we will extract title into a separate column, convert rare titles and assign them a numeric value based on table below

|Existing Title|Equivalent Title| Assigned Numeric value|
|---:|:---|:---|
|Mr|Mr|1|
|Miss, Mlle, Ms,|Miss|2|
|Mrs, Mme|Mrs|3|
|Master|Master|4|
|Lady, Countess, Capt, Col,Don, Dr, Major, Rev, Sir, Jonkheer, Dona|Rare|5|

* **Sex** : The following conversion will be applied

|Sex|Assigned numeric Value|
|---:|:---|
|Male|0|
|Female|1|

* **Age**: The age will be grouped based on table below

|Age Range|Assigned numeric value|
|---:|:---|
|<= 11 |0|
|>11 and <=18|1|
|>18 and <=22|2|
|>22 and <=27|3|
|>27 and <=33|4|
|>33 and <=40|5|
|>40 and <=66|6|
|>66|7|

* **Fare**: Fare amount will be grouped based on table below.

|Fare Amount|Assigned numeric value|
|---:|:---|
|<= 7.91 |0|
|> 7.91 and <= 14.454|1|
|> 14.454 and <= 31)|2|
|> 31 and <= 99|3|
|> 99 and <= 250|4|
|> 250|5|

## New Attributes
We will create two more additional attributes

* **Age times Class**: It was observed initially that both the age and ticket class have an impact on the survivability. 
* **Fare per Person**: Similarly we also observed the number of relatives have an impact on survivability. The fare amount is very varied, the assumption here being head of family paid for the trip of all the relatives.



# Model Generation

## Model Comparison

## Hyperparameter tuning
### Grid Search
### Optuna

# Model Interpretation
