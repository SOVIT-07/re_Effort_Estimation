🧮 RE Effort Estimation
📘 Overview

This project aims to estimate software requirement engineering effort using machine learning techniques.
It helps predict the effort required for software requirements based on project parameters (such as size, complexity, and productivity factors).

The model is implemented in Python using libraries like pandas, numpy, matplotlib, and scikit-learn.

📂 Project Structure
RE_Effort_Estimation/
│
├── Requirement_Estimation_Dataset.csv   # Dataset containing requirement parameters and effort values
├── re_effort_estimation.py              # Python script for data preprocessing, model training & evaluation
├── .gitattributes                       # Git configuration file
└── README.md                            # Project documentation (this file)

⚙️ Features
#Data loading and preprocessing
#Exploratory data analysis (EDA) with visualizations
#Linear Regression model for effort estimation
#Performance evaluation using metrics like R² score and RMSE

🚀 How to Run
1. Clone the repository
git clone https://github.com/SOVIT-07/RE_Effort_Estimation.git
cd RE_Effort_Estimation
2. Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn
3. Run the script
python re_effort_estimation.py
4.View results
Model accuracy and RMSE are printed on the console.
Graphs showing data distributions and predictions are displayed.

📊 Dataset
The dataset (Requirement_Estimation_Dataset.csv) contains features related to software requirements such as:
>Requirement Size
>Complexity
>Productivity Factor
>Estimated Effort

🧠 Model Used
>Linear Regression for continuous effort estimation
>Cross-validation and performance metrics evaluation included

📈 Sample Output
>Displays regression line and actual vs predicted effort plots
>Shows R² Score and RMSE values

📜 License
This project is licensed under the MIT License – feel free to use and modify it.
