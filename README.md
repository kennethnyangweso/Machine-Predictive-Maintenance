# 🏭 Machine Predictive Maintenance with Machine Learning

![Status](https://img.shields.io/badge/Status-Completed-success)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Machine%20Learning](https://img.shields.io/badge/ML-Binary%20Classification-orange)
![Google%20Colab](https://img.shields.io/badge/Notebook-Google%20Colab-yellow)
![License](https://img.shields.io/badge/License-MIT-green)

This project develops a **Predictive Maintenance** system to determine whether a machine will experience a **Failure** or **No Failure**. The goal is to help industries reduce downtime, improve reliability, and plan maintenance proactively using machine learning.

---

## 📌 Project Objectives
- ✔️ Build a **binary classification** model  
- ✔️ Predict **Failure vs No Failure**  
- ✔️ Identify key contributing features  
- ✔️ Support proactive maintenance decisions  

---

## 🧠 Machine Learning Workflow
- Data Cleaning & Preprocessing  
- Exploratory Data Analysis (EDA)  
- Feature Engineering  
- Model Training & Evaluation  
- Selecting the Best Performing Model  

---

## 📂 Dataset
The dataset contains machine operating parameters and failure indicators.

**Example Features**
- Temperature  
- Rotational Speed  
- Torque  
- Tool Wear  
- Other operating conditions  

**Target Variable**
- `1` → Failure  
- `0` → No Failure  

---

## 📊 Exploratory Data Analysis (EDA)

The EDA phase focused on understanding machine behavior and identifying failure patterns.

### ✔️ Key Steps
- Checked for missing values and cleaned data  
- Explored class distribution of Failure vs No Failure  
- Examined feature distributions (temperature, speed, torque, tool wear, etc.)  
- Identified relationships between variables and failure likelihood

### 📈 Visualizations

**Distribution of Target and Failure Type**

<img width="1383" height="484" alt="image" src="https://github.com/user-attachments/assets/c5c55257-c0d8-4dfd-bfcd-aaa7ca0af330" />

**Outlier Detections**

<img width="978" height="584" alt="image" src="https://github.com/user-attachments/assets/7f016c02-0da7-45d8-b90d-a762c43d92eb" />

**Pairplot of Numeric Features by Target**

<img width="1305" height="1274" alt="image" src="https://github.com/user-attachments/assets/39379f8d-6cd6-43c1-9f43-5bff75b880ed" />


### 🔍 Insights

- The 'Distribution of Target' plot shows a significant class imbalance. The number of instances with 'No Failure' (0) is much higher than those with 'Failure' (1).
- The 'Distribution of Failure Type' plot reveals that 'No Failure' is the most frequent category, which is consistent with the target distribution. Among the failure types, 'Heat Dissipation Failure' and 'Power Failure' appear to be the most common, while 'Random Failures' and 'Tool Wear Failure' are less frequent, and 'Overstrain Failure' is the least frequent.
- The boxplot clearly highlights the presence of outliers in several numerical features, particularly rotational_speed_[rpm], torque_[nm], and tool_wear_[min].
- It's difficult to see clear linear separations between the 'Failure' (orange) and 'No Failure' (blue) instances in most scatter plots, suggesting that simple linear models might not fully capture the relationships.

---

## 🧠 Modeling Approach

### 🔹 Machine Learning Models

- Logistic Regression
- Random Forest
- XGBoost
- Support Vector Machine

### 🔹 Deep Learning Models

- Artificial Neural Network

### 🔹 Ensemble Model

- Top two best performing models 


## 🏆 Model Performance
The best model  (**Ensemble model**) Random Forest and XGBoost combo demonstrated strong performance in distinguishing failure vs non-failure cases.

| Metric     | Performance |
|-----------|-------------|
| Accuracy  | 99.0%  |
| Precision | 95.5%   |
| Recall    | 87.5%  |
| F1-Score  | 91.5%  |


### 🔍 Model Performance Insights

- **High Accuracy (99.0%)** shows the model is highly reliable in predicting machine conditions overall.
- **Strong Precision (95.5%)** means when the model predicts a failure, it is correct most of the time. This reduces false alarms and prevents unnecessary maintenance actions.
- **Good Recall (87.5%)** indicates the model captures most actual failure cases, which is critical in preventing unexpected downtimes.
- **Balanced F1-Score (91.5%)** confirms a strong balance between correctly identifying failures and minimizing incorrect predictions.

Overall, the model demonstrates excellent predictive capability, making it highly suitable for real-world predictive maintenance applications. It effectively identifies failure risks while minimizing false alerts, supporting smarter maintenance planning and improved machine reliability.

### 📊 Evaluation Visuals

<img width="597" height="463" alt="image" src="https://github.com/user-attachments/assets/e934b9ef-9a56-400b-a1b6-45b4a5e508ef" />

<img width="459" height="463" alt="image" src="https://github.com/user-attachments/assets/8c3cb721-04bf-413b-8bf8-eea84441482f" />

<img width="984" height="584" alt="image" src="https://github.com/user-attachments/assets/8c438bba-1ae1-489a-82d2-557ead89dc20" />

#### Key Insights 
- The Ensemble model demonstrates very strong performance, particularly excelling in minimizing false positives while maintaining a high number of true positives and relatively low false negatives. Compared to the individual Random Forest, XGBoost, and SVM models, the Ensemble model appears to be the most balanced and effective in this classification task, especially in correctly identifying "No Failure" and having fewer critical errors (false negatives).
- The ROC-AUC curve indicates that the Ensemble model has outstanding classification performance, demonstrating a strong ability to differentiate between the classes, with a very high balance of sensitivity and specificity.
-  rotational_speed_[rpm] is by far the most important feature, with the highest mean importance value (around 0.0175).

---

## 🛠️ Tech Stack
- Python  
- Pandas  
- NumPy  
- Scikit-Learn  
- Matplotlib / Seaborn  
- Google Colab  

---

## 📊 Key Insights
- 🔹 The model effectively predicts potential machine failures  
- 🔹 Enables **preventive maintenance** strategies  
- 🔹 Helps reduce downtime and financial loss  

---

## ▶️ How to Run
1. Clone this repository  
2. Open the notebook in **Google Colab**  
3. Run all cells  
4. View predictions and evaluation metrics  

       Machine-Predictive-Maintenance/
       │
       ├── data/
       ├── notebooks/
       ├── src/
       │   ├── preprocessing.py
       │   ├── train.py
       │   ├── evaluate.py
       │   └── predict.py
       ├── models/
       ├── requirements.txt
       ├── README.md
       └── LICENSE

---

## 🚀 Future Enhancements
- Deploy as a web app or API  
- Integrate real–time streaming data  
- Explore deep learning approaches  

---
### Deployment Illustration:

  ![Screenshot (36)](https://github.com/user-attachments/assets/c4d1efaa-7ba3-46d3-8619-717972d1a323)


## **👤 Author**

**Kenneth Nyangweso**

**Data Scientist | Electrical & Telecommunications Engineer**




