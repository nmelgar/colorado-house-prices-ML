# Colorado Houses Built Before 1980
<hr>

The state of Colorado has a large portion of their residential dwelling data that is missing the year built and they would like you to build a predictive model that can classify if a house is built pre 1980.

## Background
<hr>

The clean air act of 1970 was the beginning of the end for the use of asbestos in home building. By 1976, the U.S. Environmental Protection Agency (EPA) was given authority to restrict the use of asbestos in paint. Homes built during and before this period are known to have materials with asbestos. You can [read more about this ban](https://www.asbestos.com/mesothelioma-lawyer/legislation/ban/)</br>

The state of Colorado has a large portion of their residential dwelling data that is missing the year built and they would like you to build a predictive model that can classify if a house is built pre 1980. </br>

Colorado gave you home sales data for the city of Denver from 2013 on which to train your model. They said all the column names should be descriptive enough for your modeling and that they would like you to use the latest machine learning methods. </br>


## Data
[Dwellings_Denver](https://github.com/byuidatascience/data4dwellings/raw/master/data-raw/dwellings_denver/dwellings_denver.csv) </br>
[Dwellings_ML](https://github.com/byuidatascience/data4dwellings/raw/master/data-raw/dwellings_ml/dwellings_ml.csv) </br>
[Dwellings_Neighborhoods_ML](https://github.com/byuidatascience/data4dwellings/raw/master/data-raw/dwellings_neighborhoods_ml/dwellings_neighborhoods_ml.csv) </br>
[Data Description](https://github.com/byuidatascience/data4dwellings/blob/master/data.md) </br>

## Questions and Tasks
<hr>

1. Create 2-3 charts that evaluate potential relationships between the home variables and before1980. Explain what you learn from the charts that could help a machine learning algorithm.
2. Build a classification model labeling houses as being built “before 1980” or “during or after 1980”. Your goal is to reach or exceed 90% accuracy. Explain your final model choice (algorithm, tuning parameters, etc) and describe what other models you tried.
3. Justify your classification model by discussing the most important features selected by your model. This discussion should include a chart and a description of the features.
4. Describe the quality of your classification model using 2-3 different evaluation metrics. You also need to explain how to interpret each of the evaluation metrics you use.

## Required Technologies
<hr>

+ At least Python 3.10.11
+ Pandas
+ Altair

## Author
<hr>

+ Nefi Melgar (mel16013@byui.edu)