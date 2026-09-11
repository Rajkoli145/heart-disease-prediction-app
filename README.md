# Heart Disease Prediction App (Ensemble Model with Deployment)

**Author:** Raj koli | **Roll No:** 150096724145  
**Course:** BTech ElonMusk 2428 - Machine Learning  
**Assignment:** Assignment 10 - Building Ensemble Model With Deployment  

Random Forest (Bagging Ensemble) classification model deployed using Streamlit to predict coronary heart disease risk from clinical patient parameters.

- **Dataset:** Kaggle UCI Heart Disease Dataset (`heart.csv` - 303 patient records, 13 clinical features)
- **Model:** Random Forest Classifier (`n_estimators=100`, `max_depth=6`, `oob_score=True`)
- **Test Accuracy:** 85.25%
- **Recall (Sensitivity):** 96.97% (high sensitivity ensures heart disease cases are not missed)
- **ROC-AUC Score:** 0.9286
- **5-Fold Cross-Validation Accuracy:** 81.83%

---

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Files
- `app.py` — Streamlit interactive web user interface
- `heart_disease_model.pkl` — Trained Random Forest ensemble model
- `heart.csv` — Kaggle UCI Heart Disease dataset
- `Assignment10_Ensemble_HeartDisease.ipynb` — Model training, cross-validation, and evaluation notebook
- `Assignment10_Ensemble_HeartDisease_Report.docx` — Assignment report in Microsoft Word format
- `Assignment10_Ensemble_HeartDisease_Report.pdf` — Assignment report in PDF format
- `requirements.txt` — Python dependencies
