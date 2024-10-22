**Fake News Detection System**

## Overview
The Fake News Detection System is a machine learning project aimed at identifying and classifying news articles as either fake or real. This project utilizes Natural Language Processing (NLP) techniques and a logistic regression model to predict the authenticity of news articles based on their content.

## Table of Contents
- [Technologies Used](#technologies-used)
- [Dataset](#dataset)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [How It Works](#how-it-works)
- [Acknowledgments](#acknowledgments)
- [License](#license)

## Technologies Used
- Python
- NLTK (Natural Language Toolkit)
- Scikit-learn
- Pandas
- NumPy
- TfidfVectorizer
- Random Forest Classifier
- TKINTER
- newspaper3k library

## Dataset
The dataset used for this project is a collection of news articles labeled as fake or real. The dataset can be sourced from [Kaggle] or any other reliable source that provides labeled news data. 

I have used following dataset into my model:
•	Dataset1: https://www.kaggle.com/c/fake-news/data?select=train.csv
•	Dataset2: https://www.kaggle.com/c/fake-news/data?select=test.csv


## Features
- Combines title and author information for better accuracy.
- Performs text preprocessing, including stemming and stopword removal.
- Utilizes TF-IDF for feature extraction from the text.
- Implements a random forest clasifier model for classification.
- Predicts whether a news article is real or fake based on user input.

## Installation
To run this project, you need to have Python installed on your machine. Follow the steps below to set up the environment:

1. Clone the repository:
   ```bash
   git clone https://github.com/Satyam2812/FAKE-NEWS-PREDICTION.git
   cd fake-news-detection
   ```

2. Install the required packages:
import numpy as np
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import nltk
from newspaper import Article # To extract news from URL
import tkinter as tk
from tkinter import messagebox


## Usage
1. Ensure you have the dataset `train.csv` in the project directory.
2. Run the main script:
   ```bash
   python main.py
   ```
3. Enter the news URL or content when prompted to check if it is fake or real.

## How It Works
1. **Data Preprocessing**: The data is loaded, and the content is combined from the title and author fields. Text preprocessing is performed to clean the text and prepare it for analysis.
2. **Feature Extraction**: The text data is converted into numerical format using TF-IDF vectorization.
3. **Model Training**: The dataset is split into training and testing sets. A logistic regression model is trained on the training data.
4. **Prediction**: The model predicts whether new news articles are real or fake based on the learned patterns.

## Acknowledgments
I would like to express my sincere gratitude to **Dr. Ashaq Hussain Bhatt** for his invaluable guidance and support throughout the development of this project.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

```

### Instructions for Use
1. Replace `Satyam2812` in the Git clone command with your GitHub username.
2. If you have specific installation requirements, create a `requirements.txt` file listing all necessary packages, or you can include installation commands directly in the README.
3. Adjust any sections to better fit your project, including adding more details where necessary.
