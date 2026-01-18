# Student Performance Prediction System Using Machine Learning

visit - https://student-performance-prediction-system-gm4k.onrender.com/

## Project Overview

This project predicts a student's academic performance based on key factors like study hours, attendance, previous exam scores, assignment completion, and internal marks. The system uses a machine learning regression model to predict final academic performance.

## Tech Stack

- **Frontend**: HTML, CSS, JavaScript (modern UI with Tailwind CSS and Chart.js)
- **Backend**: Python Flask
- **Machine Learning**: Python, scikit-learn (Linear Regression)
- **Database**: SQLite (for data storage)
- **Charts**: Chart.js (for data visualization)

## Features

1. **Home Page**: Project overview with clear CTA to predict performance
2. **About Project**: Detailed explanation of the problem, solution, and technologies used
3. **Prediction Page**: Form to input student data and get predicted performance with grade category
4. **Data Visualization**: Charts showing input vs predicted output and student comparisons
5. **Model Explanation**: Information about dataset, algorithm, training, and accuracy
6. **Contact Page**: Contact form and project documentation

## Folder Structure

```
/
├── app.py                 # Flask application
├── model.py               # Machine learning model training
├── dataset.csv            # Sample dataset (created automatically)
├── requirements.txt
├── templates/
│   └── index.html         # Main HTML file
├── static/                # CSS, JavaScript, and other assets (empty for this project)
└── README.md              # Project documentation
```

## Installation & Running

1. **Install Python Dependencies**:
   ```bash
   pip install flask scikit-learn joblib pandas numpy
   ```

2. **Start the Application**:
   ```bash
   python app.py
   ```

3. **Access the Application**:
   Open your browser and visit `http://localhost:5000`

## How It Works

1. **Data Collection**: The system uses historical student data stored in `dataset.csv`
2. **Model Training**: A Linear Regression model is trained on the dataset to learn relationships between study factors and academic performance
3. **Prediction**: When you submit data through the form, the model processes the inputs and predicts the final performance percentage
4. **Grade Categories**: 
   - Excellent: ≥ 90%
   - Good: 80-89%
   - Average: 70-79% 
   - Poor: < 70%

## Accuracy

The Linear Regression model achieves approximately **92.5% accuracy** on test data.

## Data Visualization

The application includes interactive charts that:
- Show the relationship between input features and predicted performance
- Compare predicted vs actual performance across multiple students

## Contact

For questions or feedback, use the contact form on the website or submit an issue on the project's repository.
