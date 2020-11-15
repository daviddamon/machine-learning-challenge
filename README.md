## Machine Learning Challenge - Exoplanet Exploration

------

### Description:

The purpose of this project was to create machine learning models capable of classifying candidate exoplanets using NASA's Kepler Space Observatory Exoplanet Search Dataset.  The dataset reviewed included a csv file of 6992 rows x 40 columns of data collected over a six year period. 

For the analysis, multiple Machine Learning and Deep Learning algorithms were used, including Random Forest, KNN (K Nearest Neighbor), and SVM (Support Vector Machines) from the Pandas Scikit-Learn modules. The Keras Pandas Neural Network and Deep Learning algorithm was also used as another method of analysis.

------

### Analysis:

- **Summary**

  For each model, the raw csv dataset was read into a Pandas data frame and cleaned to remove null columns and rows. Desired Features were selected for model training, and the Train Test Split process was run. Data was then scaled using the MinMaxScaler model, the model was trained, and initial results were produced. The models then received Hyperparameter Tuning using GridSearchCV to reach final scores that were displayed in a Classification Report. 

  The results shown below for each classification algorithm demonstrate that different algorithms have different performance characteristics when compared to each other with a Confusion Matrix.  The results of the Confusion Matrix is summarized in a Classification Report, which can be easily interpreted and compared. The f1-scores in these tables show the average scores for the model,  and are the primary numbers to compare between models. 

  As indicated in the tables below, the highest f1-score reached for Accuracy has a value of 0.90, and was obtained using the Random Forest model.  The Keras Deep Neural Network model attained an almost identical Accuracy value of 0.8987. Therefore, the Random Forest and Keras algorithms seem to be the most accurate overall.

  However, when examining only the FALSE POSITIVE criteria, both the KNN and SVM algorithm have identical highest f1-scores of 0.99. 

  The model with the highest CONFIRMED f1-score is the Random Forest with a value of 0.85.
  
  

- **Random Forest**

  With Hyperparameter Tuning using GridSearchCV:

  ```
  Best Parameters: {'max_depth': 100, 'n_estimators': 1000}
  Best Score: 0.8928053980890562
  ```

  ```
  Training Grid Score: 1.0
  Testing Grid Score: 0.8953089244851259
  ```

  Classification Report:

  ```
           precision    recall  f1-score   support
  
       CANDIDATE       0.83      0.76      0.80       411
       CONFIRMED       0.84      0.86      0.85       484
  FALSE POSITIVE       0.97      1.00      0.98       853
  
        accuracy                           0.90      1748
       macro avg       0.88      0.87      0.87      1748
    weighted avg       0.90      0.90      0.90      1748
  ```

  

- **KNN (K Nearest Neighbor)**

  Classification Report for Basic Model:

  ```
                 precision    recall  f1-score   support
  
       CANDIDATE       0.61      0.30      0.40       411
       CONFIRMED       0.58      0.84      0.69       484
  FALSE POSITIVE       0.98      0.97      0.97       853
  
        accuracy                           0.78      1748
       macro avg       0.72      0.70      0.69      1748
    weighted avg       0.78      0.78      0.76      1748
  ```

  Classification Report with Hyperparameter Tuning using GridSearchCV:

  ```
                 precision    recall  f1-score   support
  
       CANDIDATE       0.68      0.57      0.62       411
       CONFIRMED       0.68      0.75      0.71       484
  FALSE POSITIVE       0.98      1.00      0.99       853
  
        accuracy                           0.83      1748
       macro avg       0.78      0.77      0.77      1748
    weighted avg       0.83      0.83      0.83      1748
  ```

  

- **SVM (Support Vector Machines)**

  Basic Model:

  ```
  Training Data Score: 0.8455082967766546
  Testing Data Score: 0.8415331807780321
  ```

  Best Parameters and Scores:

  ```
  {'C': 10, 'gamma': 0.0001}
  0.8714435412861394
  ```

  With Hyperparameter Tuning using GridSearchCV:

  ```
  Training Grid Score: 0.8758344459279038
  Testing Grid Score: 0.8735697940503433
  ```

  Classification Report:

  ```
                  precision    recall  f1-score   support
  
       CANDIDATE       0.81      0.64      0.72       411
       CONFIRMED       0.75      0.85      0.79       484
  FALSE POSITIVE       0.98      1.00      0.99       853
  
        accuracy                           0.87      1748
       macro avg       0.84      0.83      0.83      1748
    weighted avg       0.87      0.87      0.87      1748
  ```

  

------



- **Keras Deep Learning**

  *Normal Neural Network:*

  One layer with 10 units. Epoch = 1000

  ```
  Normal Neural Network - Loss: 0.25185666367991294, Accuracy: 0.8981693387031555
  ```

  *Deep Neural Network:*

  Two layers with 12 units. Epoch = 400

  ```
Deep Neural Network - Loss: 0.2640214726090158, Accuracy: 0.8987414240837097
  ```
  
  With Keras Deep Neural Network, increasing the number of layers, nodes, and epochs beyond the numbers shown actually decreased the Accuracy. This may be due to "overfitting" the model. with additional tweaking and testing of parameters, the Keras model should be able to reach an Accuracy  score much closer to 1.0.

  

------



### Files Included:

- **data**

  exoplanet_data.csv

  

- **notebooks**

  model_Random_Forest.ipynb

  model_KNN.ipynb

  model_SVM.ipynb

  model_Keras.ipynb

  

- **saved_models**

  Random_Forest_model.sav

  KNN_model.sav

  SVM_model.sav

  neural_network_model.h5

  deep_learning_model.h5

------





