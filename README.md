# Machine Learning Projects by Sonal Sonarghare

## Project 1: Salary Predictions of Data Professions 💼

### Overview 📄
This project aims to predict the salaries of data professionals based on a dataset containing various features such as experience, job role, and performance. The accurate prediction of salaries is crucial for both job seekers and employers.

### Dataset 📊
The dataset includes the following columns:
- `FIRST NAME`: First name
- `LAST NAME`: Last name
- `SEX`: Gender
- `DOJ`: Date of joining the company
- `CURRENT DATE`: Current date of data
- `DESIGNATION`: Job role/designation
- `AGE`: Age
- `SALARY`: Target variable, the salary of the data professional
- `UNIT`: Business unit or department
- `LEAVES USED`: Number of leaves used
- `LEAVES REMAINING`: Number of leaves remaining
- `RATINGS`: Ratings or performance ratings
- `PAST EXP`: Past work experience

### Steps 🛠️

1. **Exploratory Data Analysis (EDA)**:
    - Conducted data visualization and summary statistics.
    - Identified patterns and insights from the data.

2. **Feature Engineering**:
    - Created new features and transformed existing ones to improve model performance.

3. **Data Preprocessing**:
    - Handled missing values.
    - Encoded categorical variables.
    - Scaled or normalized features as needed.

4. **Model Development**:
    - Trained various regression models: Linear Regression, Decision Trees, Random Forests, Gradient Boosting.
    - Evaluated models using RMSE, MAE, and R-squared metrics.

5. **Model Selection**:
    - Selected the best-performing model based on evaluation metrics.
    - The Linear Regression model was chosen due to its superior performance.

6. **Model Deployment**:
    - Deployed the model using ML pipelines for end-to-end machine learning processes.
    - Considered frameworks like Flask for deployment.

7. **Recommendations**:
    - Provided actionable insights and strategies to improve earnings based on the model's findings.

### Results 📈
The Linear Regression model was selected as the best performer with the following metrics:
- RMSE: 9787.76
- MAE: 4636.4
- R-squared: 0.942

### Conclusion 🎯
The project successfully predicted salaries with high accuracy, identifying significant predictors like age, past experience, and job role. Future work includes continuous monitoring and potential feature engineering for enhanced performance.

![02bfac2108763dff](https://github.com/SonalSonarghare/Mentorness/assets/116957485/d6b8a45f-9b7f-4067-8f28-00f786198dc9)
## Project 2: Fastag Fraud Detection 🚗

### Overview 📄
This project focuses on developing a machine learning solution for real-time fraud detection in FASTag transactions, an electronic toll collection system in India. The goal is to detect fraudulent transactions to prevent revenue loss.

### Dataset 📊
The dataset includes the following columns:
- `Transaction_ID`: Unique identifier for each transaction
- `Timestamp`: Date and time of the transaction
- `Vehicle_Type`: Type of vehicle involved in the transaction
- `FastagID`: Unique identifier for Fastag
- `TollBoothID`: Identifier for the toll booth
- `Lane_Type`: Type of lane used for the transaction
- `Vehicle_Dimensions`: Dimensions of the vehicle
- `Transaction_Amount`: Amount associated with the transaction
- `Amount_paid`: Amount paid for the transaction
- `Geographical_Location`: Location details of the transaction
- `Vehicle_Speed`: Speed of the vehicle during the transaction
- `Vehicle_Plate_Number`: License plate number of the vehicle
- `Fraud_indicator`: Binary indicator of fraudulent activity (target variable)

### Steps 🛠️

1. **Exploratory Data Analysis (EDA)**:
    - Conducted data visualization and summary statistics.
    - Identified patterns and insights from the data.

2. **Feature Engineering**:
    - Encoded categorical variables using Label Encoding.
    - Handled class imbalance using SMOTE.

3. **Model Development**:
    - Split data into training and testing sets (70:30).
    - Trained models: Random Forest, MLPClassifier (Neural Network), Gradient Boosting, SVC.
    - Fine-tuned models using RandomizedSearchCV and GridSearchCV.

4. **Model Evaluation**:
    - Evaluated models using Accuracy, Precision, Recall, F1 Score, and ROC-AUC Curve.

5. **Model Selection**:
    - Selected the Gradient Boosting model based on its superior performance with an accuracy of 99.92%.

6. **Real-time Fraud Detection**:
    - Deployed the model using Flask for real-time fraud detection.

7. **Explanatory Analysis**:
    - Identified the most critical features for fraud detection: Amount_paid, Transaction_Amount, and Vehicle_Dimensions.

### Results 📈
The Gradient Boosting model was selected with the following metrics:
- Accuracy: 0.9991
- Precision: 0.9984
- Recall: 1.0000
- F1 Score: 0.9992
- AUC: 1.0000

### Conclusion 🎯
The project successfully detected fraudulent transactions with near-perfect accuracy, ensuring the security and integrity of FASTag transactions. Future work includes continuous model monitoring and exploring additional features for further improvements.

![0404ce28d983874d](https://github.com/SonalSonarghare/Mentorness/assets/116957485/e598c9d5-58e3-4469-bcc0-28abc1f2aa88)

## Reel Task: Real-World Applications of Naive Bayes 🎥

### Overview 📄
This reel highlights real-world applications of the Naive Bayes algorithm, showcasing its simplicity, efficiency, and effectiveness in handling high-dimensional data. Naive Bayes is particularly useful in various domains, including:

1. **Text Classification**:
    - Text classification using Naive Bayes involves training a probabilistic model on a dataset of labeled news articles to categorize into predefined categories.
      
2. **Spam Filtering**:
    - Naive Bayes spam filtering calculates word probabilities from labeled emails to classify new emails as spam or not spam.
      
3. **Sentiment Analysis**:
   - Employed to determine the sentiment of text data, such as reviews and social media posts.Sentiment analysis using Naive Bayes involves categorizing text into positive, neutral, or negative sentiments based on the probability of words appearing in each category. 

4. **Medical Diagnosis**:
    - Disease Prediction: Medical diagnosis using Naive Bayes involves training a probabilistic model on medical data to predict diseases based on symptoms and relevant factors.

### Key Points 🌟
- **Simplicity**: Naive Bayes is easy to implement and understand.
- **Efficiency**: Performs well with large datasets and high-dimensional data.
- **Effectiveness**: Despite its simplicity, it provides robust performance in many practical applications.

### Examples 📚
1. **Spam Filtering**: Email providers like Gmail use Naive Bayes to filter out spam emails, improving user experience by reducing unwanted emails.
2. **Sentiment Analysis**: Businesses use sentiment analysis to gauge customer sentiment from reviews, helping them improve products and services.
3. **Medical Diagnosis**: In healthcare, Naive Bayes assists in predicting diseases, enabling early intervention and treatment.

### Conclusion 🎯
Naive Bayes is a powerful yet simple algorithm with diverse real-world applications, making it a valuable tool in the machine learning toolkit.

https://github.com/SonalSonarghare/Mentorness/assets/116957485/f540af2e-4d08-44e5-931e-1a6a4c717dfd

## Contact 📧
Sonal Sonarghare  
Email: [sonalsonarghare30@gmail.com](mailto:sonalsonarghare30@gmail.com)  
A.P Shah Institute of Technology


