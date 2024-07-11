# %%
# package import
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, plot_confusion_matrix

# %%
# data description
# https://github.com/byuidatascience/data4dwellings/blob/master/data.md
# this data is ml ready
# https://raw.githubusercontent.com/byuidatascience/data4dwellings/master/data-raw/dwellings_ml/dwellings_ml.csv
data_ml = pd.read_csv("data/dwellings_ml.csv")


# %%
data_ml[data_ml.yrbuilt < 1980]

# GRAND QUESTION 1
#   Create 2-3 charts that evaluate potential relationships between the home variables and before1980.
#   Explain what you learn from the charts that could help a machine learning algorithm.

# %%
# mean of live area through the years
after_1900 = data_ml.query("yrbuilt >= 1900")
mean_area_per_year = after_1900.groupby("yrbuilt")["livearea"].mean().reset_index()
mean_area_per_year_chart = px.line(
    mean_area_per_year,
    x="yrbuilt",
    y="livearea",
    # markers=True,
    labels={"yrbuilt": "Year Built", "livearea": "Live Area"},
    title="Live area mean through the years",
)
mean_area_per_year_chart.add_vline(
    x=1980,
    line_width=1,
    line_dash="dash",
    line_color="red",
    annotation_text="1980",
)
mean_area_per_year_chart.show()

# %%
# mean of bedrooms through the years
mean_bedrooms_per_year = after_1900.groupby("yrbuilt")["numbdrm"].mean().reset_index()
bedrooms_year_chart = px.scatter(
    mean_bedrooms_per_year,
    x="yrbuilt",
    y="numbdrm",
    # markers=True,
    labels={"yrbuilt": "Year Built", "numbdrm": "Number of bedrooms"},
    title="Mean of bedrooms through the years",
)
bedrooms_year_chart.add_vline(
    x=1980,
    line_width=1,
    line_dash="dash",
    line_color="red",
    annotation_text="1980",
)
bedrooms_year_chart.show()

# %%
# average price through the years
xx_century = data_ml.query("yrbuilt >= 1900 and yrbuilt < 2000")
price_per_year = xx_century.groupby("yrbuilt")["sprice"].mean().reset_index()
price_per_year_chart = px.bar(
    price_per_year,
    x="yrbuilt",
    y="sprice",
    # markers=True,
    labels={"yrbuilt": "Year Built", "sprice": "Selling Price"},
    title="Average Selling price in the XX century",
)
price_per_year_chart.add_vline(
    x=1980,
    line_width=1,
    line_dash="dash",
    line_color="red",
    annotation_text="1980",
)
price_per_year_chart.show()

# %%
# GRAND QUESTION 2
#     Build a classification model labeling houses as being built “before 1980” or “during or after 1980”.
#     Your goal is to reach or exceed 90% accuracy. Explain your final model choice
#     (algorithm, tuning parameters, etc) and describe what other models you tried.

# GRAND QUESTION 3
#     Justify your classification model by discussing the most important features selected by your model.
#     This discussion should include a feature importance chart and a description of the features.

# GRAND QUESTION 4
#     Describe the quality of your classification model using 2-3 different evaluation metrics.
#     You also need to explain how to interpret each of the evaluation metrics you use.

# %%
