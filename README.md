# Machine-Predictive-Maintenance

# **Business Understanding**

## **Overview**
Manufacturing industries rely heavily on equipment that must operate efficiently and continuously to meet production targets. Unplanned machine failures result in costly downtime, repair expenses, and lost productivity. Traditional maintenance strategies—either reactive (fix after failure) or preventive (scheduled maintenance)—often lead to inefficiencies, either by acting too late or too early.

Predictive maintenance uses historical and real-time sensor data to anticipate equipment failures before they occur, enabling timely interventions that minimize downtime and extend asset life.

## **Problem Statement**
The company currently experiences production losses due to unexpected machine failures. While sensor data is available, it is not being fully leveraged to predict when a failure is imminent.


## **Project Objectives**

### **Primary Objectives:**

1. Develop a binary classification model to predict machine failure ( No Failure OR Failure).

### **Secondary Objectives:**

1. Perform Exploratory Data Analysis (EDA) to uncover key drivers of failure.

2. Rank the importance of sensor features contributing to failure.

3. Build a real-time or batch scoring system for predictive alerts.

## **Metrics for Success**
### **Technical Metrics**

**Task	Metric	Description**

1. Binary Classification	Accuracy, Recall, F1-score,

- Evaluate prediction performance; F1 is crucial due to class imbalance.

### **Business Metrics**
For measuring real-world impact:

- Reduction in unplanned downtime (%)

- Cost savings from optimized maintenance schedules ($)
# **Data Understanding**

## **Dataset Overview**
The dataset consists of 10,000 records and 10 columns, collected from manufacturing machines equipped with various sensors. Each row represents a machine log entry with associated sensor readings and failure information.

### **Feature Summary**
1. UDI: Unique identifier for each record.
2. Product ID: Serial number identifying each machine or product.
3. Type: Category of the product or machine — typically ‘L’ (light), ‘M’ (medium), or ‘H’ (heavy) load.
4. Air temperature [K]: Ambient air temperature measured in Kelvin.
5. Process temperature [K]: Internal temperature of the machine during processing, in Kelvin.
6. Rotational speed [rpm]: Speed of the machine's rotation in revolutions per minute.
7. Torque [Nm]: Torque applied by the machine in Newton-meters.
8. Tool wear [min]: Total time (in minutes) the tool has been used, contributing to wear.
9. Target: Binary indicator of failure — 0 for no failure, 1 for failure.
10. Failure Type: Categorical label of failure cause (e.g., Tool Wear, Overstrain, Power Failure, etc.).

# **Data Preparation**

The process involved:
- Dropping of unnecessary columns such as product ID
- Adding new features such as; temperature difference
- Data preprocessing i.e label encoding and scaling of the features

## **Exploratory Data Analysis**

![image](https://github.com/user-attachments/assets/0339ff12-5e44-4a83-89ab-cc3660e73155)

**Observations**
- The distribution is clearly right-skewed (positively skewed). This means the tail of the distribution extends further to the right.
- The peak of the distribution, which represents the most frequent rotational speeds, appears to be around 1450-1500 RPM.

![image](https://github.com/user-attachments/assets/b5f497d7-a696-4db0-90aa-3c7b3fcc7654)

**Observations**
- The 'Distribution of Target' plot shows a significant class imbalance. The number of instances with 'No Failure' (0) is much higher than those with 'Failure' (1). This imbalance will need to be addressed during model training (e.g., using techniques like oversampling, undersampling, or using appropriate evaluation metrics like F1-score).

- The 'Distribution of Failure Type' plot reveals that 'No Failure' is the most frequent category, which is consistent with the target distribution. Among the failure types, 'Heat Dissipation Failure' and 'Power Failure' appear to be the most common, while 'Random Failures' and 'Tool Wear Failure' are less frequent, and 'Overstrain Failure' is the least frequent. This imbalance in failure types also needs to be considered for the multiclass classification task.

![image](https://github.com/user-attachments/assets/2e9882fc-48a3-41eb-a4f8-6588e61c8b01)

**Observations**

- This pairplot helps visualize relationships between numerical features, colored by the 'target' variable (Failure or No Failure).

- It's difficult to see clear linear separations between the 'Failure' (orange) and 'No Failure' (blue) instances in most scatter plots, suggesting that simple linear models might not fully capture the relationships.

- Some features, like tool_wear_[min], show some separation or different distributions based on the target, indicating they might be important predictors.

- The diagonal KDE plots show the distribution of each feature for both target classes. This can reveal if the distribution of a feature is significantly different for failed vs. non-failed instances.

# Modeling

For the modeling part we will build FOUR traditional models that are good with classification. These models are:
- Logistic Regression
- Random Forest Classifier
- XGBoost
- Support Vector Machine

For the deep learning models we will try Artificial Neural Network (ANN)

**Best Model**
- Ensemble model( votingclassifier for the random forest and xgboost)
- The best performing models were the random forest classifier and XGBoost

# Evaluating the best model

Ensemble Classification Report:

              precision    recall  f1-score   support

           0       0.99      1.00      1.00      1939
           1       0.92      0.75      0.83        61

    accuracy                           0.99      2000

![image](https://github.com/user-attachments/assets/2f5723eb-aa9f-47fc-9bfe-d77aa8007969)

**Ensemble Model Classification Report insights**

**a) Recall**

1. Class 0 (Normal): 1.00
→ The ensemble model correctly identified all normal/no-failure cases.

2. Class 1 (Failure): 0.75
→ A significant improvement over the SVM (0.57) and slightly better than XGBoost (0.74). The model correctly identified 75% of actual failures, which is solid considering the imbalance.
**b) F1-Score**

1. Class 0: 1.00
→ Perfect balance between precision and recall for the majority class.

2. Class 1: 0.83
→ This is the highest F1-score for the failure class among all models so far (Random Forest: 0.82, XGBoost: 0.80, SVM: 0.66). This suggests the ensemble provides the best overall balance for failure detection.

**c) Accuracy**

Overall accuracy is 99%, but more importantly, it achieves this while maintaining a relatively high recall and F1-score for the failure class.
→ This makes it the most balanced and effective model for detecting both normal and failure cases in your binary setup.

**Ensemble Model Confusion Matrix observations**

- True Negatives (No Failure correctly predicted as No Failure): 1935 instances. This is the highest number of correctly predicted "No Failure" instances among all models reviewed.
T- rue Positives (Failure correctly predicted as Failure): 46 instances. This is tied with the Random Forest model for the highest number of correctly predicted "Failure" instances.
- False Positives (No Failure incorrectly predicted as Failure): 4 instances. This is the lowest number of false positives among all models reviewed, indicating very few instances where "No Failure" was wrongly flagged as "Failure."
- False Negatives (Failure incorrectly predicted as No Failure): 15 instances. This is tied with the Random Forest model for the lowest number of false negatives, meaning it missed fewer actual "Failure" events.

Overall Comparison: The Ensemble model demonstrates very strong performance, particularly excelling in minimizing false positives while maintaining a high number of true positives and relatively low false negatives. Compared to the individual Random Forest, XGBoost, and SVM models, the Ensemble model appears to be the most balanced and effective in this classification task, especially in correctly identifying "No Failure" and having fewer critical errors (false negatives).

![image](https://github.com/user-attachments/assets/75ebf8f2-5c45-4a4d-b97e-5014a3fe22e6)

**ROC-AUC observations**

- A high AUC (close to 1.0) indicates that the model is very good at distinguishing between the positive and negative classes.
- The steepness of the curve at the beginning suggests that the model can achieve a high true positive rate (TPR) with a very low false positive rate (FPR). This means it's very effective at identifying actual positive cases without incorrectly flagging too many negative cases.
- The fact that the curve stays near the top (TPR = 1.0) for a large portion of the FPR range further confirms its excellent discriminatory power.

**Overall Impression:** The ROC-AUC curve indicates that the Ensemble model has outstanding classification performance, demonstrating a strong ability to differentiate between the classes, with a very high balance of sensitivity and specificity.

![image](https://github.com/user-attachments/assets/3026bb3b-3751-43cf-a7f8-48d684e16288)

**Observations**

1. **Dominant Features:**

- rotational_speed_[rpm] is by far the most important feature, with the highest mean importance value (around 0.0175).
- temp_diff and torque_speed_interaction are also highly important, following rotational_speed_[rpm] closely.
- tool_wear_[min] and torque_[nm] are the next most important features, though with a noticeable drop in importance compared to the top three.

2. **Less Important Features (among the top 10):**
- type and process_temperature_[k] have very low mean importance values, almost negligible compared to the top features.
- air_temperature_[k] even shows a slightly negative mean importance, which can sometimes happen with permutation importance for very unimportant features due to randomness, suggesting it might not be relevant or could even be noise.

Overall: The plot clearly shows that a few features (rotational_speed_[rpm], temp_diff, torque_speed_interaction) are overwhelmingly more influential on the Ensemble model's predictions than the others. This information is crucial for understanding which inputs drive the model's decisions and for potential feature selection or engineering.

# **Conclusions**

1. Ensemble Model Outperforms Neural Network
The ensemble model delivered superior performance across key metrics, especially for the minority class (failure):

- Recall (0.75): Higher than the neural network's (0.61), indicating better detection of failure events.

- F1-Score (0.83): Indicates a better balance between precision and recall.

- Accuracy (~99%): Maintained strong overall accuracy without sacrificing minority class detection.

- ROC-AUC: 98%: Demonstrates the model's strong ability to distinguish between failure and no-failure cases.

2. Effective Failure Detection with Fewer Errors
- The ensemble model produced fewer false positives and false negatives, making it more reliable in a predictive maintenance setting where missing a failure (false negative) could be costly or dangerous.

3. Neural Network Struggles with Class Imbalance
- Despite good performance on the majority class, the neural network underperformed on the failure class, making it less suitable when accurate fault detection is crucial.


# **Recommendations**

1. Adopt the Ensemble Model for Deployment

- Its consistent performance, particularly on minority (failure) cases, makes it ideal for real-world use where catching rare failures is critical.

2. Monitor for Model Drift

- Regularly retrain and validate the model with updated data to ensure continued accuracy, especially if failure patterns evolve.

3. Further Improvements 

- Explore cost-sensitive learning or custom loss functions that penalize false negatives more heavily if needed.

- Use techniques like feature selection or domain-specific feature engineering to further refine model performance.

4. Implement Model Explainability

- Add explainability tools (e.g., SHAP, LIME) to understand which features drive failure predictions—useful for both maintenance teams and regulatory compliance.

# **Deployment**

1. Save  the Trained Model

- Save the model or keep it in memory for immediate use in Colab.

2. Define a Prediction Function

- Create a function that accepts input data, processes it and returns a predicted failure type.

3. Set Up Gradio Interface

- Define input components matching the model’s features (e.g., temperature, speed, torque).

- Set the output to return the predicted failure type.

4. Launch Gradio in Colab

- Use share=True in launch() so a public URL is generated.

- Gradio will automatically detect Colab and host the interface temporarily.

5. Test the Interface

- Open the generated Gradio *.gradio.live URL.

- Manually input values or test with sample input to verify predictions.

6. Deploy Permanently

- For long-term hosting, deploy the app to Hugging Face Spaces using gradio deploy.

# Deployment Illustration:

  ![Screenshot (36)](https://github.com/user-attachments/assets/c4d1efaa-7ba3-46d3-8619-717972d1a323)
