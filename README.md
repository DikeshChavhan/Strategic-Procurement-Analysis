🧠 Procurement Strategy Optimization Using Kraljic Matrix & Machine Learning

📘 Overview

This project leverages Machine Learning and Strategic Procurement Analysis to classify products and materials based on the Kraljic Matrix Framework — a powerful model for supply chain strategy optimization.

By analyzing supply risk, profit impact, and other procurement parameters, this model automatically identifies whether an item is Strategic, Leverage, Bottleneck, or Non-Critical, enabling smarter sourcing and inventory decisions.
--
🎯 Problem Statement

Procurement professionals often face challenges in determining which items deserve strategic attention and which can be managed operationally. Traditional classification methods are manual, time-consuming, and prone to human bias.

This project aims to:

Analyze procurement data using risk and impact-based metrics.

Build an automated machine learning classifier that categorizes items into Kraljic quadrants.

Identify the most influential features driving procurement strategy.

Provide visual insights to support data-driven supply chain decisions.
---
📊 Dataset Description

Dataset: realistic_kraljic_dataset.csv
Records: 1,000 procurement entries
Purpose: Simulated real-world data representing sourcing, cost, risk, and sustainability attributes.

Feature	Description
Product_ID	Unique product identifier
Product_Name	Material or item name
Supplier_Region	Supplier’s geographical location
Lead_Time_Days	Average number of delivery days
Order_Volume_Units	Quantity ordered regularly
Cost_per_Unit	Cost of procuring one unit
Supply_Risk_Score	Rating (1–5) for supply uncertainty
Profit_Impact_Score	Rating (1–5) for profit influence
Environmental_Impact	Sustainability impact score
Single_Source_Risk	Whether sourced from a single supplier (Yes/No)
Kraljic_Category	Target category (Strategic, Leverage, Bottleneck, Non-Critical)
----
⚙️ Methodology

Data Preprocessing

Cleaned missing and inconsistent entries.

Encoded categorical data.

Normalized numeric features for uniform scaling.

Exploratory Data Analysis (EDA)

Distribution plots for numerical attributes.

Correlation matrix to detect feature relationships.

Visual mapping of Supply Risk vs Profit Impact quadrants.

Model Development

Applied Random Forest Classifier for robust, non-linear decision-making.

Split dataset (75% train / 25% test).

Used LabelEncoder and StandardScaler for preprocessing.

Model persisted using joblib for future inference.

Evaluation Metrics

Accuracy, Precision, Recall, F1-score, and Confusion Matrix.

Achieved 100% accuracy (due to synthetic consistency).
---
🧩 Kraljic Matrix Categories
Category	Description	Strategy
Strategic Items	High risk, high profit impact	Build partnerships, long-term planning
Leverage Items	Low risk, high profit impact	Maximize negotiation power
Bottleneck Items	High risk, low profit impact	Reduce dependency, find alternatives
Non-Critical Items	Low risk, low profit impact	Optimize efficiency and automation
---
📈 Results & Insights

Top 5 influential features:

Profit Impact Score

Cost per Unit

Supply Risk Score

Order Volume Units

Lead Time Days

The model successfully classifies new procurement items with exceptional precision.

Visualization of the Kraljic Matrix confirms clear quadrant separation between items.
----

🧠 Technologies Used
Category	Tools/Packages
Programming Language	Python
Data Analysis	Pandas, NumPy
Visualization	Matplotlib, Seaborn
Machine Learning	Scikit-learn
Model Storage	Joblib
Notebook Environment	Jupyter Notebook
