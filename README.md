# Student-Management-System
# 🎓 Student Management System using XGBoost & SMOTE

This project predicts student performance levels (High, Medium, Low) using machine learning on educational behavioral data.  
It also includes an Object-Oriented Student Simulator to showcase Python OOP skills.
---

## 📊 Dataset Overview

The dataset contains 17 features about students' behavior, demographics, and academic engagement.  
It is stored in a CSV file named: `xAPI-Edu-Data.csv`

### 🧾 Column Names

- `gender`  
- `NationalITy`  
- `PlaceofBirth`  
- `StageID`  
- `GradeID`  
- `SectionID`  
- `Topic`  
- `Semester`  
- `Relation`  
- `raisedhands`  
- `VisITedResources`  
- `AnnouncementsView`  
- `Discussion`  
- `ParentAnsweringSurvey`  
- `ParentschoolSatisfaction`  
- `StudentAbsenceDays`  
- `Class` (Target variable: High, Medium, Low)

---

## Model Used

- Algorithm: XGBoost Classifier  
- Class Balancing:** SMOTE (Synthetic Minority Oversampling Technique)

### Model Parameters

```python
XGBClassifier(
    n_estimators=300,
    learning_rate=0.08,
    max_depth=5,
    eval_metric='mlogloss',
    use_label_encoder=False,
    random_state=42
)
```

---

## 📈 Visualizations Included

- Class distribution before and after SMOTE  
- Confusion matrix  
- ROC curves  
- Precision-Recall curves  
- Top feature importances using Seaborn and Matplotlib  

These make the project visually impressive for resumes or GitHub!

---

## Object-Oriented Student Simulation

Besides machine learning, this project includes a Python class called `Student` that simulates:

- Enrollments and attendance  
- Subject-wise grading  
- Parent and teacher messages  
- Summary of attendance, performance, and grades  

This demonstrates real-world OOP (Object-Oriented Programming) usage alongside data science.

---

## Main Libraries Used

- `pandas`  
- `numpy`  
- `seaborn`  
- `matplotlib`  
- `scikit-learn`  
- `xgboost`  
- `imblearn` (for SMOTE)

---

## Requirements

Install all dependencies using:

```bash
pip install -r requirements.txt
```

---

## 🚀 How to Run

1. Clone or download this repository  
2. Make sure `xAPI-Edu-Data.csv` is in the same folder as the Python file  
3. Install requirements with `pip install -r requirements.txt`  
4. Run the main Python script using VS Code or terminal

---

## Highlights

- Achieved 76% accuracy on test data  
- Balanced the dataset using SMOTE  
- Clean, beautiful visualizations  
- Simulates students with real-life behavior  
- Perfect for showcasing Machine Learning + Python OOP

---

## 📬 Contact

Feel free to reach out for feedback !

---
