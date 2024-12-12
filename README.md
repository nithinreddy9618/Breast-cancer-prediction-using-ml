# Breast-cancer-prediction-using-ml
2. Problem Statement
Breast cancer is a major global health concern, affecting millions of individuals each year. Early and accurate diagnosis is essential for effective treatment and improving survival rates. However, the complex nature of medical data, including feature interactions and variability in patient demographics, makes the prediction of breast cancer challenging.
Traditional probabilistic models like Naive Bayes are simple, interpretable, and often perform well on small datasets but may struggle with complex patterns. On the other hand, advanced techniques like Support Vector Machine (SVM) are effective in handling nonlinear relationships and separating data into high-dimensional spaces.
This study focuses on comparing the performance of Naive Bayes and SVM in predicting breast cancer, leveraging data preprocessing, feature engineering, and hyperparameter optimization to enhance the prediction accuracy.
________________________________________
3. Objective
The primary objective of this study is to develop and evaluate predictive models for breast cancer diagnosis using two machine learning techniques: Naive Bayes and SVM. The study aims to Assess the accuracy and robustness of both models on breast cancer data, Highlight the role of preprocessing and feature engineering in improving model performance, Optimize hyperparameters and perform an ablation study to determine the impact of features and configurations on the predictions.
The insights gained will contribute to building more reliable and efficient diagnostic tools for healthcare professionals.
4. Proposed Method
4.1. Workflow: Enhancing Breast Cancer Prediction
1.	Dataset Collection
Collect datasets containing breast cancer features and corresponding diagnostic outcomes. Examples include:
o	UCI Machine Learning Repository: Wisconsin Breast Cancer Dataset.
o	Kaggle: Breast Cancer Diagnostic Dataset.
These datasets typically include features such as cell radius, texture, perimeter, area, smoothness, compactness, and malignancy labels.
2.	Data Preprocessing
o	Handling Missing Values: Missing values are imputed using statistical techniques like mean or median.
o	Standardization: Features are scaled to have a mean of 0 and a standard deviation of 1, crucial for SVM's performance.
o	Normalization: Features are scaled between 0 and 1 for uniformity across variables.
3.	Data Visualization
Employ visualization techniques to explore relationships among features and outcomes:
o	Scatter plots to identify trends between features.
o	Heatmaps to reveal feature correlations.
o	Histograms to assess feature distributions.
4.	Feature Engineering
o	Feature Selection: Use correlation analysis and mutual information to retain the most predictive features.
o	Feature Transformation: Apply transformations to reduce skewness and improve linear separability.
5.	Model Building
o	Naive Bayes Classifier: A probabilistic model leveraging Bayes' theorem, particularly Gaussian Naive Bayes for continuous data.
o	Support Vector Machine (SVM): A robust classifier with various kernels (linear, RBF) to handle complex patterns.


6.	Hyperparameter Tuning
o	Naive Bayes: Adjust smoothing parameter (e.g., Laplace smoothing).
o	SVM: Optimize hyperparameters like kernel type, C (regularization), and gamma.
7.	Performance Evaluation
Evaluate models using metrics such as:
o	Accuracy: Proportion of correct predictions.
o	Precision, Recall, F1-Score: Measure class-wise performance.
o	ROC-AUC Score: Assess classification thresholds.
4.2. Dataset Collection
For this study, we utilized breast cancer datasets containing diagnostic indicators and corresponding labels. Examples:
•	UCI Machine Learning Repository: Wisconsin Breast Cancer Dataset.
UCI Machine Learning Repository breast-cancer-wisconsin-dataset
•	Kaggle: Breast Cancer Diagnostic Data.
These datasets provide a robust foundation for analysing breast cancer prediction.
________________________________________
4.3. Data Preprocessing
•	Handling Missing Values: Missing data was imputed using the mean for continuous variables.
•	Standardization: Features were standardized for better performance in SVM.
•	Normalization: Applied Min-Max scaling to ensure consistency across variables.
________________________________________
4.4. Data Visualization
Visualization techniques provided insights into data structure and feature relationships:
1.	Scatter plot of radius mean vs. texture mean.
 
2.	Correlation heatmap of feature relationships.
 
3.	Pair plot malignant vs. benign cases.
 

4.	Distribution plot of malignant vs. benign cases.
 
________________________________________
4.5. Machine Learning Algorithms
1.	Naive Bayes Classifier:
A simple probabilistic model based on Bayes' theorem, assuming feature independence. It is effective for small datasets but may struggle with overlapping feature distributions.
2.	Support Vector Machine (SVM):
A robust model for classification tasks, utilizing kernels (linear, RBF) to map data into higher-dimensional spaces for better separability. SVM excels at handling nonlinear patterns.


5. Results & Discussion
5.1. Naive Bayes Classifier
Naive Bayes performed well with small datasets and achieved reasonable accuracy. However, it was less effective at capturing complex feature interactions, leading to higher misclassification rates.
5.2. Support Vector Machine
SVM outperformed Naive Bayes, particularly with optimized hyperparameters and nonlinear kernels. It captured complex relationships in the dataset, achieving higher accuracy and a better ROC-AUC score.
5.3. Comparison and Justification
•	SVM outperformed Naive Bayes in all metrics, demonstrating its effectiveness in handling complex datasets.
•	The superior performance of SVM justifies its suitability for breast cancer prediction, especially when data preprocessing and hyperparameter tuning are applied.

 

The comparison plot illustrating the accuracy of Naïve Bayes and SVM classifiers based on accuracy, precision, f1-score and recall values. It shows that SVM classifier has more accuracy, highlighting its superiority in predicting malignant and benign
6. Conclusion
This study demonstrated that SVM, with its ability to capture nonlinear relationships and its adaptability to hyperparameter tuning, outperforms Naive Bayes in breast cancer prediction tasks. Leveraging advanced models and preprocessing techniques can lead to significant improvements in diagnostic accuracy, contributing to better healthcare outcomes.
________________________________________
7. References
1.	Lichman, M. (2013). UCI Machine Learning Repository.
2.	Cortes, C., & Vapnik, V. (1995). Support-Vector Networks.
3.	Brownlee, J. (2020). Machine Learning Mastery.
4.	Hastie, T., Tibshirani, R., & Friedman, J. (2009). The Elements of Statistical Learning.
________________________________________
8. Contributors
Name	      Responsibility
[22691A05F2, F4]-       	    Data Collection & Preprocessing
[22691A05F3, F6]-	      Model Development & Testing
[22691A05F1, F5]-	      Report Writing & Visualization
________________________________________
8.Code Availability
GitHub Link: github repository link

