# Machine Learning-Driven Clinical Risk Stratification for Lymphedema

This repository contains the official Python implementation and data science pipeline for the prognostic analysis of lymphedema risks based on clinical demographics, tumor profiles, and treatment burdens.

📢 **Note on Publication Status:** This study is currently **under peer review** in a scientific journal. The repository will be updated with the official citation, DOI link, and supplementary materials immediately upon formal publication.

---

## 🔬 Project Overview
Lymphedema remains a significant secondary complication following breast cancer interventions. This project leverages advanced machine learning methodologies (including ensemble strategies and stacking meta-learners) alongside interpretability frameworks (SHAP) to stratify patient risk profiles. 

The pipeline segments analysis into distinct clinical components:
* Multi-modality treatment burden scoring.
* Synergistic impact of surgery (ALND vs. SLNB) and radiotherapy.
* Molecular subtype risk mapping (St. Gallen Criteria).
* Non-linear threshold modeling for removed lymph nodes.

---

## 📁 Repository Structure & File Guide

Below is the roadmap of the codebase. Each script is self-contained and handles a specific domain of the clinical analysis:

| File Name | Description | Primary Output Artifact |
| :--- | :--- | :--- |
| `clinical_ml_pipeline.py` | Core engine executing cross-validated GridSearch on 9+ base models and stacking meta-learners. | `all_model_metrics.csv`, `top_performers_AUC.png` |
| `treatment_burden_analysis.py` | Calculates cumulative risk score based on treatment intensity (ALND, Chemo, RT). | `treatment_burden_detailed_risks.png` |
| `surgery_rt_synergy_plot.py` | Evaluates the compound risk interaction specifically between axillary surgery and radiotherapy. | `2_surgery_rt_synergy.png` |
| `molecular_subtype_risk_analysis.py` | Classifies patients using the St. Gallen Criteria and analyzes lymphedema prevalence across biological phenotypes. | `1_molecular_subtype_risk.png` |
| `clinical_risk_matrix.py` | Generates interaction risk matrices for baseline patient demographics (Age vs. BMI). | `risk_matrix_bmi_age.png` |
| `surgical_bmi_risk_matrix.py` | Cross-examines body mass index cohorts with surgical aggressiveness levels. | `risk_matrix_english.png` |
| `bmi_risk_logistic_regression.py` | Fits logistic regression curves to map continuous BMI risk progression. | `2_bmi_surgical_threshold.png` |
| `lymph_node_risk_curve.py` | Models non-linear probability thresholds relative to the count of dissected lymph nodes. | `lymphedema_risk_analysis_plot.png` |
| `feature_importance_comparison.py` | Conducts dual-layered feature profiling using Recursive Feature Elimination (RFE) and Global SHAP metrics. | `combined_shap_rfe_analysis.png` |
