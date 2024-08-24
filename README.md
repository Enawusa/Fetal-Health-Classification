# Fetal-Health-Classification

1. # **ABSTRACT**
 
The study investigates the classification of fetal health using machine learning techniques. The objective is to predict whether a fetus is in a normal, suspect, or pathological state based on data collected from cardiotocographic (CTG) tests. Various machine learning models are employed, and their performance is evaluated to determine the most effective approach for this classification task.

3. # **INTRODUCTION**

Fetal health monitoring is a crucial aspect of prenatal care, aimed at detecting potential health issues that could affect both the mother and the child. Cardiotocography (CTG) is a common non-invasive method used to monitor fetal well-being by recording fetal heart rate (FHR) and uterine contractions. The data from these tests can be complex and challenging to interpret, which is where machine learning comes into play. This study aims to apply and compare different machine learning models to automate the classification of fetal health, potentially aiding in quicker and more accurate decision-making in clinical settings.

5. # **DATA DESCRIPTION**


The dataset used in this study contains several features derived from CTG tests. These features include measurements such as baseline fetal heart rate, accelerations, decelerations, and uterine contractions. The data is labeled into three categories: Normal, Suspect, and Pathological, which represent the health status of the fetus. Before analysis, the data is preprocessed to handle any missing values, outliers, or inconsistencies. Feature selection techniques are also applied to reduce dimensionality and improve model performance.

7. # **Data Analysis**


Exploratory Data Analysis (EDA) is conducted to gain insights into the distribution and relationships of the features in the dataset. Visualizations such as histograms, box plots, and correlation matrices are used to understand how different features correlate with each other and how they relate to the fetal health categories. Statistical tests are also performed to identify which features have the most significant impact on the classification of fetal health. This step helps in selecting the most relevant features for the machine learning models.

9. #  **MODEL USED AND ANALYSIS**

   
Several machine learning models are implemented to classify fetal health, including Decision Trees, Random Forests, Support Vector Machines (SVM), and Neural Networks. Each model is trained and tested on the dataset, and their performance is evaluated using metrics such as accuracy, precision, recall, and F1-score. The models are also compared in terms of their ability to generalize to new, unseen data. Cross-validation is used to ensure the reliability of the results. The analysis reveals that some models perform better than others, with ensemble methods like Random Forests generally providing higher accuracy and robustness.

11. # **CONCLUSION**


The study concludes that machine learning models can effectively classify fetal health based on CTG data. The best-performing models demonstrate high accuracy and could potentially be used in clinical settings to assist healthcare professionals in monitoring fetal well-being. The research highlights the importance of data preprocessing and feature selection in improving model performance. Future work could involve the integration of these models into real-time monitoring systems and further validation with larger, more diverse datasets.
