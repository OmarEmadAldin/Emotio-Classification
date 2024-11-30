# Human Pose Recognition and Classification

This project demonstrates a pipeline for detecting human poses using MediaPipe Holistic, collecting and labeling pose data, and building a machine learning classification model. The trained model is serialized with pickle for deployment in various applications.

## Features

-  **Pose Detection**: Leverages MediaPipe Holistic to detect human poses.
-  **Data Collection**: Saves pose coordinates and labels into a .csv file.
-  **Data Processing**: Uses pandas to handle and split data into training, validation, and test sets.
-  **Model Training**: Tests multiple machine learning pipelines to find the best classifier.
-  **Model Deployment**: Serializes the optimal trained model with pickle for reuse.

## Dependencies
The following libraries are required to run the project:

- mediapipe
- numpy
- pandas
- sklearn
- pickle

Install these packages using:
    ```bash
pip install mediapipe numpy pandas scikit-learn
```

## Project Workflow
1. Pose Detection and Data Collection

- Use MediaPipe Holistic to extract pose landmarks from images or videos.
- Save the extracted landmarks with their corresponding labels in a .csv file.

2. Data Preparation

- Load the .csv file with pandas Split the data into training validation, and test datasets.

3. Model Training

- Test the following machine learning pipelines:
```bash

        Logistic Regression
        Ridge Classifier
        Random Forest Classifier
        Gradient Boosting Classifier
```

-Standardize features with StandardScaler.
-Evaluate each model to determine the best performer.

4. Model Deployment

- Save the trained model to a .pkl file using pickle.

5. Usage
- Load the serialized model with:
```python
import pickle

with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

```
- Use the model for prediction:
```python

prediction = model.predict(data)
```
## File Structure

- body_lang_recog.ipynb: Jupyter Notebook containing the code for data collection, training, and saving the model.
- coord.csv: Folder to store collected pose data (.csv files).
- body_language.pkl: Folder to save trained models (.pkl files).
