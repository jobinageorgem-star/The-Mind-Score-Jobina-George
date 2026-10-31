THE MIND SCORE - PHYSICAL HEALTH ANALYSIS: scripts
Jobina George, 24EMDT7149

Setup:  pip install numpy pandas scipy scikit-learn xgboost
        (XGBoost 3.4.1 or later; rerun nhanes_prediction.py to refresh XGBoost results)

analysis/  (run in this order, in one folder)
 1. merge_nhanes.py        .xpt files -> nhanes_analysis.csv (one row per participant)
 2. validate_nhanes.py     data checks on nhanes_analysis.csv
 3. nhanes_survey_run.py   survey design, weighted prevalence, first odds ratios
 4. nhanes_table1.py       weighted characteristics by PHQ-9 group, trend tests
 5. nhanes_models.py       survey-weighted logistic Models 1-4
 6. nhanes_prediction.py   four prediction models with/without PHQ-9, CV, AUC, Brier, importance,
                           plus weighted accuracy/sensitivity/specificity/precision/F1 (new)
    Usage for 2-6: python <script>.py nhanes_analysis.csv

calculator/
 export_models_for_web.py  exports fitted models to models.json
 build.py                  builds index.html from calculator.template.html + models.json + predict.js
 predict.js                in-browser prediction code
 test_parity.py / .js      compare Python and JavaScript predictions
 index.html                the finished single-page calculator
