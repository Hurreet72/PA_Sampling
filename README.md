# Sampling of data (Credit Card Fraud Detection) 

This project studies how different sampling methods change the performance of machine learning models when the original credit card dataset is highly imbalanced.


## Discussion

In this experiment, my main goal was to understand how sampling techniques help machine learning models handle a highly imbalanced credit card fraud dataset.
Initially, the dataset contained far more legitimate transactions than fraudulent ones. Because of this imbalance, models tend to get biased toward the majority class and may fail to learn the patterns of fraud properly. To solve this, I first created a **balanced dataset** using undersampling (selecting an equal number of majority and minority class records). After balancing, I generated **five different training samples** using five sampling techniques and tested each of them on **five machine learning models**.
To keep the comparison fair, I used a **single common holdout test set** for evaluation. This ensured that every sampling–model combination was tested on the same data, so differences in accuracy were due to the sampling method and model behavior rather than different test splits.
From the results, it was clear that the choice of sampling method can noticeably affect performance. Some models performed consistently well across multiple sampling techniques, while others were sensitive to how the training sample was created. In general, ensemble-based models such as Random Forest / Gradient Boosting handled the sampled datasets more robustly because they can capture non-linear relationships and reduce overfitting using multiple trees. On the other hand, distance-based and margin-based models (like KNN and SVM) were more dependent on how representative the sample was, and their accuracy changed more across sampling methods.



## Methodology

1. Loaded the dataset directly from the GitHub raw CSV link.
2. Identified the target column (`Class`) and checked class imbalance.
3. Balanced the dataset using random undersampling.
4. Created five samples using:
   - **Sampling1:** Simple Random Sampling  
   - **Sampling2:** Systematic Sampling  
   - **Sampling3:** Stratified Sampling  
   - **Sampling4:** Bootstrap Sampling  
   - **Sampling5:** Cluster Sampling  
5. Trained five models:
   - Logistic Regression  
   - Random Forest  
   - SVM (RBF)  
   - KNN  
   - Gradient Boosting  
6. Evaluated all combinations using accuracy and stored results in a 5×5 table.
7. Selected:
   - Best sampling technique for each model
   - Best overall model + sampling combination


## Results
  
<img width="1580" height="478" alt="sampling" src="https://github.com/user-attachments/assets/99a0b30b-3f2d-4997-99e8-1ff6acf49ae8" />
<img width="1980" height="678" alt="table" src="https://github.com/user-attachments/assets/11e0bae2-92ce-4f3e-88ba-b498821f405d" />



## Result Graph
To visualize performance differences, I used a graph to compare accuracy across sampling methods and models :

<img width="1580" height="777" alt="graph" src="https://github.com/user-attachments/assets/dbcebca3-5088-41d1-b553-f3c4c63df16f" />
