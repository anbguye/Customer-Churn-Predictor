# Customer Churn Prediction

This project is an interactive web application built with Streamlit for predicting customer churn using machine learning models. It leverages a dataset of bank customer information to train and deploy models that assess the likelihood of a customer leaving the bank. The app provides predictions, explanations, and automated email generation to aid in customer retention.

## Features

- **Interactive Prediction Interface**: Select a customer from the dataset and input or modify their details to get churn predictions.
- **Multiple Model Ensemble**: Uses XGBoost, Random Forest, and K-Nearest Neighbors models for robust predictions, displaying individual and average probabilities.
- **AI-Powered Explanations**: Integrates with OpenAI's Groq API to generate natural language explanations for predictions based on customer data and feature importances.
- **Automated Email Generation**: Creates personalized retention emails with incentives tailored to the customer's profile.
- **Data Analysis**: The project includes exploratory data analysis in the Jupyter notebook with visualizations for understanding churn patterns.

## Models Used

The application loads pre-trained models saved as pickle files:

- XGBoost (`xgb_model.pkl`)
- Random Forest (`rf_model.pkl`)
- K-Nearest Neighbors (`knn_model.pkl`)
- Additional models trained: Decision Tree, SVM, Naive Bayes, Voting Classifier, and variants with feature engineering and SMOTE.

Models were trained on the churn dataset with features like Credit Score, Age, Tenure, Balance, Number of Products, etc., after preprocessing including one-hot encoding for categorical variables.

## Data

The project uses the `churn.csv` dataset, which contains customer information such as:
- Customer ID, Surname
- Credit Score, Geography, Gender, Age
- Tenure, Balance, Number of Products
- Has Credit Card, Is Active Member, Estimated Salary
- Exited (target variable: 1 for churned, 0 otherwise)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/anbguye/Customer-Churn-Predictor.git
   cd Customer-Churn-Predictor
   ```

2. Install dependencies using Poetry:
   ```bash
   poetry install
   ```

   Or using pip:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up environment variables:
   - Obtain a Groq API key from [Groq](https://groq.com/).
   - Set the `GROQ_API_KEY` environment variable.

## Usage

1. Run the Streamlit app:
   ```bash
   streamlit run main.py
   ```

2. Open the app in your browser (usually at `http://localhost:8501`).

3. Select a customer from the dropdown.

4. Adjust input fields if needed and view the prediction results, including:
   - Model probabilities
   - Average churn probability
   - Explanation of the prediction
   - Generated retention email

## Project Structure

- `main.py`: Main Streamlit application script.
- `utils.py`: Utility functions for creating charts.
- `Bank_Churn.ipynb`: Jupyter notebook for data analysis, model training, and evaluation.
- `churn.csv`: Dataset file.
- `*.pkl`: Pickled machine learning models.
- `pyproject.toml`: Project dependencies and configuration.
- `README.md`: This file.

## Dependencies

- Python >= 3.10
- Streamlit
- Pandas
- NumPy
- OpenAI
- Plotly
- Scikit-learn (for models)
- XGBoost
- Imbalanced-learn (for SMOTE)

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request.

## License

This project is licensed under the MIT License.
