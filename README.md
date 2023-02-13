# Credit Risk Analysis

## Overview: 
Utilize the dataset given from LendingClub to pereform different machine learning models inorder to comapre which one would reduce bias the most to help identify credit card risk.  Knowing that this is an unbalanced dataset with 0.5% risky loans the challange is to utilize imbalanced-learn and scikit-learn libraries to build and evaluate models using resampling. 

The models used to predict credit risk are:
  * RandomOverSampler and SMOTE to oversample the data.
  * ClusterCentroids to undersample the data. 
  * SMOTEEN algorithm using an combination under/oversample approach.
  * BalancedRandomForestClassifier and EasyEnsembleClassifier to reduce bias.
  
## Results: 
  
  Inorder for the model builing to begin we first had to load the dataset and clean the data by dropping all null values and loans with a status of "Issued". The target column was then converted into "low_risk" and "high_risk" which gave us 68,470 "low_risk" loans and 347 "high_risk" loans (0.5% of the current dataset). 
  
<img width="203" alt="risk total" src="https://user-images.githubusercontent.com/111904266/218520542-f99ef30b-4105-418e-bfa4-74c1b84fdcfc.png">


#### Naive Random Oversampling Results: 
  * Counter = low_risk: 51352, high_risk: 51352
  * Balance Accuracy Score = 65%
  * Confusion Matrix =   
  [ 53,    34]   
  [ 5263, 11855]
  * Precision Score: High Risk = 0.01, Low Risk = 1.00
  * Recall (Sensitivity) Score: High Risk = .61, Low Risk = .69
  * F1 Score: High Risk = .02, Low Risk = .82  
  
  <img width="671" alt="Naive Random Oversampling results" src="https://user-images.githubusercontent.com/111904266/218526098-39b9e6e3-7f4a-40c8-91dc-f4fccc189971.png">
  
  
 #### Smote Oversampling Results: 
  * Counter = low_risk: 51352, high_risk: 51352
  * Balance Accuracy Score = 62%
  * Confusion Matrix =   
  [ 53,    34]  
  [ 6210, 10908]
  * Precision Score: High Risk = 0.01, Low Risk = 1.00
  * Recall (Sensitivity) Score: High Risk = .61, Low Risk = .64
  * F1 Score: High Risk = .02, Low Risk = .78  
  
  <img width="677" alt="SMOTE oversampling results" src="https://user-images.githubusercontent.com/111904266/218525944-3ea8f5bf-305e-4aba-a87e-9e8ce08c5cad.png">


 #### Cluster Centroids Undersampling Results: 
 * Counter = low_risk: 260, high_risk: 260
  * Balance Accuracy Score = 62%
  * Confusion Matrix =   
  [  49,   38]  
  [9066, 8052]
  * Precision Score: High Risk = 0.01, Low Risk = 1.00
  * Recall (Sensitivity) Score: High Risk = .56, Low Risk = .47
  * F1 Score: High Risk = .01, Low Risk = .64  
  
  <img width="686" alt="ClusterCentroids undersampling results" src="https://user-images.githubusercontent.com/111904266/218526015-9df658cf-1b2e-4d85-bd0d-7101bc79c00e.png">

  
  #### SMOTEENN (Over/Unersampling) Results: 
  * Counter = low_risk: 62011, high_risk: 68460
  * Balance Accuracy Score = 52%
  * Confusion Matrix =   
  [  61,   26]  
  [7127, 9991]
  * Precision Score: High Risk = 0.01, Low Risk = 1.00
  * Recall (Sensitivity) Score: High Risk = .70, Low Risk = .58
  * F1 Score: High Risk = .02, Low Risk = .74  
  
  <img width="682" alt="smoteenn (combination) results" src="https://user-images.githubusercontent.com/111904266/218526206-0167b39b-e170-4ff4-993e-258750c8f7ea.png">

  
  #### Random Forest Results:  
  * Balance Accuracy Score = 67%
  * Confusion Matrix =   
  [   30,    57]  
  [   11, 17107]
  * Precision Score: High Risk = 0.73, Low Risk = 1.00
  * Recall (Sensitivity) Score: High Risk = .34, Low Risk = 1.00
  * F1 Score: High Risk = .47, Low Risk = 1.00  
  
  <img width="664" alt="random forest results" src="https://user-images.githubusercontent.com/111904266/218526284-7d65825b-c78c-4719-8e3e-bb6dd50a0eca.png">

  
  #### Easy Ensemble AdaBoost Results: 
  * Balance Accuracy Score = 93%
  * Confusion Matrix =   
  [   79,     8]  
  [  979, 16139]
  * Precision Score: High Risk = 0.07, Low Risk = 1.00
  * Recall (Sensitivity) Score: High Risk = .91, Low Risk = .94
  * F1 Score: High Risk = .14, Low Risk = .97
  
  <img width="677" alt="EasyEnsemble AdaBoost results" src="https://user-images.githubusercontent.com/111904266/218526316-19508c10-cf3c-48f5-8f21-8eff17bae122.png">


## Analysis: 

<img width="748" alt="overall results" src="https://user-images.githubusercontent.com/111904266/218526424-33b7d25a-1e34-4ee5-b3de-1e2d78500606.png">

After comparing the results with the goal of preventing false posititves (predicting "Low Risk" when it is actually "High Risk"), my recomendation is to use the Easy Ensemble AdaBoost Results. With an accuracy score of 92% and the heightest Recall (sensitivity) rate for High Risk with .91. Looking at the Confusion Matrix there were only eight predicted Low Risk when actually High Risk. Downfall to this is it does have a high number (979) loans of predicted High Risk with an actual Low Risk.  As this false negative may casue more work for humans to find the actual 79 High Risk loans that were predicted.   

My very close second recomendation would be to use the Random Forest model based on the 67% accuracy (which is not the hightest accuracy score) but does have the highest precision for High Risk with .73 and by far the highest F1 Score for High Risk. Looking at the Confusion Matrix it had the most accurate predictions of Low Risk with actual Low Risk, it also has far fewer predicted High Risk with actual Low Risk to sort through. The downfall it did predict having 57 Low Risk with an actual High Risk. 
