import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

print("=" * 60)
print("STUDENT PERFORMANCE ANALYSIS")
print("=" * 60)

# =====================================
# LOAD DATASET
# =====================================

students = pd.read_csv("data/StudentsPerformance.csv")

print("\nDataset Loaded Successfully!\n")

print("Dataset Shape:")
print(students.shape)

print("\nColumn Names:")
print(students.columns.tolist())

print("\nFirst Five Records:")
print(students.head())
# =====================================
# DATA EXPLORATION & CLEANING
# =====================================

print("\n" + "=" * 60)
print("DATA EXPLORATION & CLEANING")
print("=" * 60)

# Dataset Information
print("\nDataset Information:\n")
students.info()

# Missing Values
print("\nMissing Values:\n")
print(students.isnull().sum())

# Duplicate Rows
print("\nDuplicate Rows:")
print(students.duplicated().sum())

# Data Types
print("\nData Types:\n")
print(students.dtypes)

# Statistical Summary
print("\nStatistical Summary:\n")
print(students.describe())

# Unique Values in Categorical Columns
print("\nUnique Values in Categorical Columns:\n")

categorical_columns = [
    "gender",
    "race/ethnicity",
    "parental level of education",
    "lunch",
    "test preparation course"
]

for column in categorical_columns:

    print(f"\n{column}")

    print(students[column].unique())
    # =====================================
# DATASET SUMMARY
# =====================================

print("\n" + "=" * 60)
print("DATASET SUMMARY")
print("=" * 60)

print("""
1. The dataset contains 1000 student records with 8 attributes.
2. It includes demographic details, parental education, lunch type,
   test preparation status, and scores in Math, Reading, and Writing.
3. No missing values or duplicate records were found.
4. The average scores are approximately 66 in Math, 69 in Reading,
   and 68 in Writing.
5. This dataset is suitable for analyzing factors that influence
   student academic performance.
""")
# =====================================
# FEATURE ENGINEERING
# =====================================

students["total score"] = (
    students["math score"] +
    students["reading score"] +
    students["writing score"]
)

students["average score"] = (
    students["total score"] / 3
)

students["result"] = np.where(
    students["average score"] >= 50,
    "Pass",
    "At Risk"
)

print("\nFeature Engineering Completed!")

print("\nFirst Five Records:\n")

print(
    students[
        [
            "math score",
            "reading score",
            "writing score",
            "total score",
            "average score",
            "result"
        ]
    ].head()
)
# =====================================
# QUESTION 1
# PARENTAL EDUCATION VS SCORES
# =====================================

print("\n" + "=" * 60)
print("QUESTION 1")
print("Does parental education level affect scores?")
print("=" * 60)

parent_education = (
    students.groupby("parental level of education")[
        ["math score", "reading score", "writing score"]
    ]
    .mean()
    .round(2)
)

print("\nAverage Scores by Parental Education Level:\n")
print(parent_education)
# =====================================
# VISUALIZATION 1
# BOX PLOT
# =====================================

plt.figure(figsize=(12,6))

sns.boxplot(
    data=students,
    x="parental level of education",
    y="average score"
)

plt.title(
    "Average Scores by Parental Education Level"
)

plt.xlabel(
    "Parental Level of Education"
)

plt.ylabel(
    "Average Score"
)

plt.xticks(rotation=20)

plt.tight_layout()

plt.savefig(
    "dashboard/boxplot_parent_education.png",
    dpi=300
)

plt.close()
# =====================================
# QUESTION 2
# TEST PREPARATION
# =====================================

print("\n" + "=" * 60)
print("QUESTION 2")
print("Do students who complete test preparation score higher?")
print("=" * 60)

test_prep = (
    students.groupby("test preparation course")[
        ["math score", "reading score", "writing score"]
    ]
    .mean()
    .round(2)
)

print("\nAverage Scores:\n")
print(test_prep)
# =====================================
# BAR CHART
# TEST PREPARATION COMPARISON
# =====================================

test_prep.plot(
    kind="bar",
    figsize=(10,6)
)

plt.title(
    "Average Scores by Test Preparation Course"
)

plt.xlabel(
    "Test Preparation Status"
)

plt.ylabel(
    "Average Score"
)

plt.xticks(rotation=0)

plt.legend(title="Subject")

plt.tight_layout()

plt.savefig(
    "dashboard/test_prep_bar.png",
    dpi=300
)

plt.close()
# =====================================
# QUESTION 3
# CORRELATION
# =====================================

print("\n" + "=" * 60)
print("QUESTION 3")
print("Correlation Between Subjects")
print("=" * 60)

correlation = students[
    [
        "math score",
        "reading score",
        "writing score"
    ]
].corr()

print(correlation)
# =====================================
# HEATMAP
# =====================================

plt.figure(figsize=(6,5))

sns.heatmap(
    correlation,
    annot=True,
    cmap="YlGnBu"
)

plt.title(
    "Correlation Between Student Scores"
)

plt.tight_layout()

plt.savefig(
    "dashboard/correlation_heatmap.png",
    dpi=300
)

plt.close()
# =====================================
# QUESTION 4
# GENDER PERFORMANCE
# =====================================

print("\n" + "=" * 60)
print("QUESTION 4")
print("Gender Performance")
print("=" * 60)

gender_scores = (
    students.groupby("gender")[
        [
            "math score",
            "reading score",
            "writing score"
        ]
    ]
    .mean()
    .round(2)
)

print(gender_scores)
# =====================================
# GROUPED BAR CHART
# =====================================

gender_scores.plot(
    kind="bar",
    figsize=(8,6)
)

plt.title(
    "Average Subject Scores by Gender"
)

plt.xlabel(
    "Gender"
)

plt.ylabel(
    "Average Score"
)

plt.xticks(rotation=0)

plt.legend(title="Subject")

plt.tight_layout()

plt.savefig(
    "dashboard/gender_subject_bar.png",
    dpi=300
)

plt.close()
# =====================================
# QUESTION 5
# TOTAL SCORE DISTRIBUTION
# =====================================

print("\n" + "=" * 60)
print("QUESTION 5")
print("Distribution of Total Scores")
print("=" * 60)

print(students["total score"].describe())
# =====================================
# HISTOGRAM
# =====================================

plt.figure(figsize=(10,6))

plt.hist(
    students["total score"],
    bins=20,
    edgecolor="black"
)

plt.title(
    "Distribution of Total Scores"
)

plt.xlabel(
    "Total Score"
)

plt.ylabel(
    "Number of Students"
)

plt.tight_layout()

plt.savefig(
    "dashboard/total_score_histogram.png",
    dpi=300
)

plt.close()
# =====================================
# SCATTER PLOT
# =====================================

plt.figure(figsize=(8,6))

plt.scatter(
    students["reading score"],
    students["math score"],
    alpha=0.7
)

plt.title(
    "Reading Score vs Math Score"
)

plt.xlabel(
    "Reading Score"
)

plt.ylabel(
    "Math Score"
)

plt.tight_layout()

plt.savefig(
    "dashboard/reading_vs_math_scatter.png",
    dpi=300
)

plt.close()

print("\nAll 6 visualizations have been saved successfully in the dashboard folder.")
# =====================================
# AT-RISK STUDENT SEGMENTATION
# =====================================

print("\n" + "=" * 60)
print("AT-RISK STUDENT SEGMENTATION")
print("=" * 60)

# Student is at-risk if any subject score is below 50

at_risk_students = students[
    (students["math score"] < 50) |
    (students["reading score"] < 50) |
    (students["writing score"] < 50)
]

print("\nNumber of At-Risk Students:")
print(len(at_risk_students))

print("\nPercentage of At-Risk Students:")
print(round((len(at_risk_students) / len(students)) * 100, 2), "%")

print("\nAt-Risk Students by Gender:")
print(at_risk_students["gender"].value_counts())

print("\nAt-Risk Students by Race/Ethnicity:")
print(at_risk_students["race/ethnicity"].value_counts())

print("\nAt-Risk Students by Parental Education:")
print(
    at_risk_students[
        "parental level of education"
    ].value_counts()
)