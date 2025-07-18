# 🏡 House Price Prediction using Linear Regression

This project implements a Linear Regression model using Python to predict house prices based on square footage, number of bedrooms, and number of bathrooms.

## 📌 Dataset

The dataset is from Kaggle's [House Prices - Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data), specifically using the `train.csv` file.

## 📈 Features Used

- `GrLivArea` - Above grade (ground) living area in square feet
- `BedroomAbvGr` - Number of bedrooms above ground
- `FullBath` - Number of full bathrooms
- `SalePrice` - Target variable (house price)

## 🛠️ Technologies Used

- Python 3.x
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn

## 🧠 Model

We use **Linear Regression** from `scikit-learn` to train the model.

## 🔧 Installation & Setup

1. Clone the repository or download the project folder.
2. Create and activate a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # For macOS/Linux
   .\venv\Scripts\activate   # For Windows
Install the dependencies:

bash
Copy code
pip install -r requirements.txt
Download train.csv from the Kaggle dataset and place it in the project directory.

Run the script:

bash
Copy code
python model.py
📊 Output
R² Score: Indicates how well the model explains variance in house prices.

Mean Squared Error (MSE): Measures average squared difference between actual and predicted values.

A scatter plot is generated comparing actual vs predicted house prices.

📁 File Structure
Copy code
house_price_prediction/
├── model.py
├── train.csv
├── README.md
└── requirements.txt
✨ Future Enhancements
Add more features like YearBuilt, GarageArea, etc.

Use other models like Ridge, Lasso, or Random Forest.

Save the model using joblib or pickle.

Deploy as a web app using Flask or Streamlit.

📌 Author
Archee Sinha
2nd Year B.Tech CSE (AI) Student
📫 Feel free to connect on LinkedIn
