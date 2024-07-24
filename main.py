# %%
# package import
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


# %%
# data description
# https://github.com/byuidatascience/data4dwellings/blob/master/data.md
# this data is ml ready
data_url = "https://raw.githubusercontent.com/byuidatascience/data4dwellings/master/data-raw/dwellings_ml/dwellings_ml.csv"
data_ml = pd.read_csv(data_url)

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

# %%
# drop column to avoid overfitting
data_ml.drop(columns=["yrbuilt"], inplace=True)

# %%
# define features (X) and target variable (y)
X = data_ml.drop(
    columns=[
        "before1980",
        "parcel",
    ]
)
y = data_ml["before1980"]

# split data, training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# train logistic regression model
logreg_model = LogisticRegression(max_iter=1000)
logreg_model.fit(X_train_scaled, y_train)

# train random forest classifier model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train)

# model evaluation
y_pred_logreg = logreg_model.predict(X_test_scaled)
y_pred_rf = rf_model.predict(X_test_scaled)

accuracy_logreg = accuracy_score(y_test, y_pred_logreg)
accuracy_rf = accuracy_score(y_test, y_pred_rf)

print(f"Logistic Regression Accuracy: {accuracy_logreg:.2f}")
print(f"Random Forest Accuracy: {accuracy_rf:.2f}")

# classification reports comparisson
print("\nLogistic Regression Classification Report:")
print(
    classification_report(
        y_test, y_pred_logreg, target_names=["After 1980", "Before 1980"]
    )
)

print("\nRandom Forest Classification Report:")
print(
    classification_report(y_test, y_pred_rf, target_names=["After 1980", "Before 1980"])
)

# GRAND QUESTION 3
#     Justify your classification model by discussing the most important features selected by your model.
#     This discussion should include a feature importance chart and a description of the features.
# %%
# get feature importance from the random forest model
feature_importance = rf_model.feature_importances_

importance_df = pd.DataFrame({"Feature": X.columns, "Importance": feature_importance})
importance_df = importance_df.sort_values(by="Importance", ascending=False)
importance_df = importance_df.head(10)

# create chart to display feature importance from the model
feature_importance_chart = px.bar(
    importance_df,
    x="Importance",
    y="Feature",
    title="Feature Importance from Random Forest Model",
)

feature_importance_chart.show()

# GRAND QUESTION 4
#     Describe the quality of your classification model using 2-3 different evaluation metrics.
#     You also need to explain how to interpret each of the evaluation metrics you use.

# %%
