<img width="1125" height="620" alt="image" src="https://github.com/user-attachments/assets/a7518618-40f2-4ba2-9746-ce8283bddade" /># AutoML Engine
AutoML Engine is a powerful, user-friendly tool built with Streamlit, designed to automate the model selection, training, and evaluation process for machine learning tasks, all while providing powerful features like automated feature engineering and fast model training. This tool is perfect for users looking to rapidly experiment with a variety of models, tune them using cross-validation, and select the best-performing one based on accuracy.

## Features
- **Automated Model Selection**: Automatically detects whether the task is a classification or regression problem, and provides relevant models for selection.
- **Multiple Models Training**: Train multiple models simultaneously and compare their performance using cross-validation.
- **Feature Engineering**: Automatically handles preprocessing, including standard scaling for numerical features and one-hot encoding for categorical features.
- **Fast Training & Evaluation**: Leverages efficient algorithms and parallelized model training to speed up the training process, even when multiple models are being evaluated.
- **Visualized Results**: View performance comparisons of all trained models in a bar chart for easy model selection.
- **Exporting Models**: Easily export the trained models as .pkl files for future use or deployment.
- **No Machine Learning Experience Required**: Perfect for users without AI/ML or Python expertise. The entire process is simplified and automated, making machine learning accessible to everyone.
## Workflow & UI
**1. Upload Dataset**
Enables the user to upload the CSV file on which the model will be trained.


**2. Target Column & Machine Learning Model Selection**
Allows the user to select the target column to be predicted or classified.

**3. Models Results Comparison Graph**
A bar graph illustrating the comparison of cross-validation mean scores across the trained models.

**4. Final Model Deployment**
Users can choose which model to deploy. Two deployment options are available:
1. Upload a CSV file to test the selected model and view the results.
2. Export the trained model as a pickle file for reuse in other software or programs.
