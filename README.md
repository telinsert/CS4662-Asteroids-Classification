# NASA Asteroids Classification: Predicting Planetary Hazards 

**A Machine Learning pipeline designed to classify Near-Earth Objects (NEOs) and mitigate planetary risk by maximizing hazard recall.**

---

## Project Overview
Classifying Near-Earth Objects is a high-stakes task where a single **False Negative**—mislabeling a hazardous asteroid as safe—poses a catastrophic risk. This project utilizes machine learning to process high-dimensional orbital mechanics and telemetry data from NASA to accurately identify cosmic threats.

Our primary objective was to build a robust defense model that overcomes the dataset's severe class imbalance by **optimizing for Recall**, ensuring that true threats are not missed by the algorithm.

## The Team
* **Edwin Rojas** - Data Preprocessing & Exploratory Data Analysis (EDA)
* **Edward Garcia** - Machine Learning & Support Vector Machine (SVM) Optimization
* **Joseph Lam** - Neural Network Architecture & Deep Learning Tuning

## The Dataset
The data is sourced from the **NASA JPL Near Earth Object Web Service (NeoWs)** (via [Kaggle](https://www.kaggle.com/datasets/shrutimehta/nasa-asteroids-classification/data)). 
* **Instances:** ~4,687 observations
* **Features:** Physical metrics (estimated diameters), close approach data (relative velocity, miss distance), and orbital elements.
* **Target Variable:** `Hazardous` (Boolean: True/False)
* **The Challenge:** The dataset suffers from a severe class imbalance, with only **~16%** of the asteroids classified as genuinely hazardous. 

## Repository Structure
```text
cs4662-asteroids-classification/
│
├── data/
│   └── nasa.csv                      # Raw NASA NeoWs dataset
│
├── notebooks/
│   ├── EDA_and_Preprocessing.ipynb   # Data cleaning, unit consolidation, and scaling
│   ├── SVM_Tuning.ipynb              # Baseline model training and GridSearch CV
│   ├── ANN_Implementation.ipynb      # Deep learning architecture and dropout tuning
│   └── Final_Evaluation.ipynb        # ROC curve comparisons and Feature Importance
│
├── requirements.txt                  # Python dependencies
├── .gitignore
└── README.md                         # Project documentation
