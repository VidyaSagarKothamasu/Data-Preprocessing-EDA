import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

st.set_page_config(page_title="Student Performance Dashboard", layout="wide")
st.title("Student Performance Dashboard")

# Load dataset
def load_data():
    df = pd.read_csv("Students Performance Dataset.csv")
    df.columns = df.columns.str.strip()
    if "Parent_Education_Level" in df.columns:
        df.drop(columns=["Parent_Education_Level"], inplace=True)
    return df

df = load_data()

# Sidebar
st.sidebar.header("Dashboard Controls")
show_missing = st.sidebar.checkbox("Show Missing Values", value=True)
show_heatmap = st.sidebar.checkbox("Show Correlation Heatmap", value=True)
show_study_vs_quiz = st.sidebar.checkbox("Show Study Time vs Quiz Scores", value=True)
show_distraction = st.sidebar.checkbox("Show Distraction vs Focus", value=True)
show_pattern = st.sidebar.checkbox("Show Study Pattern Distribution", value=True)
show_dept_summary = st.sidebar.checkbox("Show Department Score Summary", value=True)
show_regression = st.sidebar.checkbox("Show Linear Regression Demo", value=True)

# Missing values
if show_missing:
    st.subheader("Missing Values in Each Column")
    st.write(df.isnull().sum())
    st.subheader("Percentage of Missing Values")
    st.write((df.isnull().sum() / len(df) * 100).round(2))

# Correlation heatmap
if show_heatmap:
    st.subheader("Correlation Heatmap")
    numeric_df = df.select_dtypes(include=["int64", "float64"])
    corr = numeric_df.corr()
    fig, ax = plt.subplots(figsize=(10,6))
    sns.heatmap(corr, annot=True, cmap='coolwarm', linewidths=0.5, fmt=".2f", ax=ax)
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    st.pyplot(fig)

# Study Time vs Quiz Scores
if show_study_vs_quiz:
    st.subheader("Study Time vs Quiz Scores")
    study = np.arange(1,9)
    quiz = [65, 70, 75, 82, 85, 88, 90, 92]
    fig, ax = plt.subplots()
    ax.plot(study, quiz, marker='o')
    ax.set_xlabel("Study Time (Hours)")
    ax.set_ylabel("Quiz Score (%)")
    ax.set_title("Study Time vs. Quiz Scores")
    st.pyplot(fig)

# Distraction vs Focus
if show_distraction:
    st.subheader("Distraction vs Focus")
    distraction = ["Low", "Medium", "High"]
    performance = [88, 75, 65]
    fig, ax = plt.subplots()
    ax.bar(distraction, performance, color=["green","orange","red"])
    ax.set_ylabel("Average Quiz Score")
    ax.set_title("Distraction vs Focus")
    st.pyplot(fig)

# Study Pattern Distribution
if show_pattern:
    st.subheader("Study Pattern Distribution")
    labels = ["Morning","Afternoon","Evening","Night"]
    sizes = [25,30,30,15]
    fig, ax = plt.subplots()
    ax.pie(sizes, labels=labels, autopct='%1.1f%%', wedgeprops=dict(width=0.4))
    ax.set_title("Study Pattern Distribution")
    st.pyplot(fig)

# Department-wise score summary
if show_dept_summary:
    st.subheader("Department-wise Score Summary")
    if "Department" in df.columns and "Total_Score" in df.columns:
        temp = df.groupby("Department")["Total_Score"].agg(["max", "mean", "min"])
        st.write(temp)
    else:
        st.write("Department or Total_Score column not found.")

# Linear Regression Demo
if show_regression:
    st.subheader("Linear Regression Demo")
    x = [5, 7, 8, 7, 2, 17, 2, 9, 4, 11, 12, 9, 6]
    y = [99, 86, 87, 88, 111, 86, 103, 87, 94, 78, 77, 85, 86]
    slope, intercept, r, p, std_err = stats.linregress(x, y)
    def myfunc(x):
        return slope * x + intercept
    mymodel = list(map(myfunc, x))
    fig, ax = plt.subplots()
    ax.scatter(x, y)
    ax.plot(x, mymodel)
    ax.set_title("Simple Linear Regression")
    st.pyplot(fig)
    st.write(f"R value: {r:.2f}, Std Error: {std_err:.2f}")

# Model Training Example
st.subheader("Final Score Prediction (Linear Regression)")
if "Department" in df.columns and "Midterm_Score" in df.columns and "Final_Score" in df.columns:
    X = pd.get_dummies(df[['Department', 'Midterm_Score']], drop_first=True)
    y = df['Final_Score']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    st.write(f"R2 Score: {r2_score(y_test, y_pred):.2f}")
    st.write(f"MSE: {mean_squared_error(y_test, y_pred):.2f}")
else:
    st.write("Required columns for model training not found.")
