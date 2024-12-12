
###  **Mean Squared Error (MSE)**
   - **Purpose**: Measures the average squared difference between actual values  and predicted values.
   - **Properties**:
     - Penalizes large errors more than small ones due to squaring.
     - Unit: Same as the square of the target variable's unit.
   - **Usage**: Helps identify models with large prediction errors but is sensitive to outliers.

###  **Root Mean Squared Error (RMSE)**
   - **Purpose**: Provides a measure of error in the same unit as the target variable, making it easier to interpret.
   - **Properties**:
     - Also sensitive to large errors (inherits this from MSE).
   - **Usage**: Often preferred for practical interpretability in terms of the target variable.


###  **Mean Absolute Error (MAE)**
   - **Purpose**: Measures the average absolute difference between actual and predicted values.
   - **Properties**:
     - Treats all errors equally, making it less sensitive to outliers compared to MSE.
     - Unit: Same as the target variable.
   - **Usage**: Useful when you want to measure the average magnitude of errors regardless of direction.


###  **R-squared (R²)**
   - **Purpose**: Represents the proportion of variance in the actual data that is captured by the model.
       - \(1\): Perfect fit (all predictions are correct).
       - \(0\): Model performs as well as the mean of \(y_true\).
       - Negative: Model performs worse than the mean of \(y_true\).
   - **Usage**: Indicates how well the model explains variability in the data.





## linear regression performance [(model found on ther internet along with the dataset)](https://www.kaggle.com/datasets/alokevil/housing/data)
    MSE: 7209418409.8793 
    RMSE: 84908.2941 
    MAE: 63562.4973 
    R²: 0.4471



## FNN model performance  (implemented using pytorch) after 10000 epochs(can vary a bit since early stop to training is implemented)
    MSE: 7006553088.0000 
    RMSE: 83705.1557 
    MAE: 62612.9766 
    R²: 0.4627