# 🏥 EHR Predictive Modeling: Mortality & Length-of-Stay

## 📌 Overview
This repository presents a comprehensive end-to-end pipeline for **predicting in-hospital mortality** and **length of hospital stay (LOS)** using Electronic Health Record (EHR) data.

The project systematically compares:
- Traditional Machine Learning models
- Sequential Deep Learning models (LSTM, Transformer)
- Clinical Large Language Models (LLMs)

The dataset is a synthetic EHR dataset inspired by MIMIC-III/IV, consisting of 5,000 patients across multiple relational tables.

---

## 📂 Repository Structure

```
├── data/
│   ├── patients_table.csv
│   ├── patient_admissions.csv
│   └── disease_diagnosis_code.csv
│
├── Notebooks/
│   ├── Preprocessing/
│   │   └── EHR_featureEngineering.ipynb
│   │
│   ├── Traditional_ML/
│   │   └── EHR_modelDevelopment.ipynb
│   │
│   ├── LSTM_Transformer/
│   │   └── LSTM_transformer.ipynb
│   │
│   └── LLM/
│       ├── llm_internist_7b.ipynb
│       └── LLM_Bio_ClinicalBERT_Modified.ipynb
│
├── scripts/
│   ├── ehr_featureengineering.py
│   └── ehr_modeldevelopment.py 
│   └── llm_internist_7b.py
│   └── llm_bio_clinicalbert_modified.py
│
└── README.md
```

---

## 📊 Dataset Description

The dataset consists of three relational tables:

### 1. Patients Table
- Demographics: age, gender, year group
- Death information

### 2. Admissions Table
- Admission/discharge timestamps
- Admission type, insurance, race
- Target variable: `hospital_expire_flag`

### 3. Diagnosis Table
- ICD-9/ICD-10 diagnosis codes
- Sequential clinical history per admission

---

## 🎯 Objectives

- Predict **in-hospital mortality** (binary classification)
- Predict **length of stay (LOS)** (regression + classification)
- Evaluate **model performance across ethnic groups**
- Compare performance across different model architectures

---
## 🧭 Workflow Overview

The project follows a structured pipeline from raw EHR data to advanced predictive modeling:

### 1. 🔧 Data Preprocessing
**Notebook:** `Notebooks/Preprocessing/EHR_featureEngineering.ipynb` 
**Script:** `Scripts/EHR_featureEngineering.py`  
- Data integration across relational tables using `subject_id` and `hadm_id`  
- ICD-9 → ICD-10 harmonization  
- Handling missing values and leakage prevention  

#### 🔹 Feature Engineering

- **Temporal Features**
  - Length of Stay (LOS) from admission and discharge time  
  - Admission year, month, and time-based patterns  

- **Demographic Features**
  - Age at admission (derived from anchor variables)  
  - Gender encoding  
  - Race and ethnicity encoding  

- **Diagnosis-Based Features**
  - ICD code normalization and mapping  
  - Grouping using Clinical Classification Software (CCSR)  
  - Multi-label binarization of diagnosis categories  
  - Number of diagnosis categories per admission  

- **Categorical Encoding**
  - Admission type  
  - Insurance type  
  - Admission and discharge locations

---

### 2. 📊 Traditional Machine Learning
**Notebook:** `Notebooks/Traditional_ML/EHR_modelDevelopment.ipynb`  
**Script:** `Scripts/EHR_modelDevelopment.py` 
- Models: Logistic Regression, Random Forest, XGBoost, LightGBM  
- Tabular feature-based modeling  
- Baseline performance benchmarking. Best for structured tabular data. Providing best performance for clinical EHR data outperforming even clinical LLM models.

---

### 3. 🔄 Sequential Deep Learning
**Notebook:** `Notebooks/LSTM_Transformer/LSTM_transformer.ipynb`  
- Patient trajectory modeling using sequential data  
- Models: LSTM, Transformer  
- Capturing temporal dependencies in admissions. 

---

### 4. 🧠 Clinical Large Language Models (LLMs)
**Notebooks:**  
- `Notebooks/LLM/llm_internist_7b.ipynb`  
- `Notebooks/LLM/LLM_Bio_ClinicalBERT_Modified.ipynb`  

**Scripts:** 
- `Scripts/llm_internist_7b.py`
- `Scripts/LLM_Bio_ClinicalBERT_Modified.py`

- Conversion of structured EHR data into clinical text  
- Models: BioClinicalBERT, Internist-7B  
- Fine-tuning and multitask learning (mortality + LOS).  

---

## 📈 Evaluation Metrics

### Classification (Mortality)
- F1-score
- AUROC
- AUPRC
- Precision & Recall
- 95% Confidence Interval (CI)

### Regression (LOS)
- MAE
- RMSE
- R²

### Additional
- Calibration (ECE, Brier Score)
- Fairness Analysis: Subgroup performance across race/ethnicity  

### 5. 📈 Evaluation & Analysis
- Metrics: AUROC, F1-score, Precision, Recall  
- Regression: MAE, RMSE, R²  
- Calibration: ECE, Brier Score  
- Fairness: Subgroup performance across race/ethnicity  

---

## 🧪 Key Findings

- Traditional ML (XGBoost) achieved strongest overall performance
- Sequential models captured temporal signals but did not outperform tabular models consistently
- Clinical LLMs showed strong discrimination but calibration issues
- Model performance varied across demographic subgroups

---

## 🚀 How to Run

### 1. Clone repository
```
git clone https://github.com/your-username/repo-name.git
cd repo-name
```

### 2. Install dependencies
```
pip install -r requirements.txt
```

### 3. Run notebooks
- Feature Engineering → `EHR_featureEngineering.ipynb`
- ML Models → `EHR_modelDevelopment.ipynb`
- Deep Learning → `LSTM_Transformer.ipynb`
- LLM Models → `llm_internist_7b.ipynb`, `LLM_Bio_ClinicalBERT_Modified.ipynb`


### 4. Run scripts (Alternative of "Run notebooks")
- Feature Engineering → `EHR_featureEngineering.py`
- ML Models → `EHR_modelDevelopment.py`
- LLM Models → `llm_internist_7b.py`, `LLM_Bio_ClinicalBERT_Modified.py`


---

## 🧠 Technologies Used

- Python (Pandas, NumPy, Scikit-learn)
- PyTorch
- HuggingFace Transformers
- XGBoost / LightGBM
- PEFT (LoRA, QLoRA)

---

### ⚠️ Disclaimer

This project uses a synthetic dataset inspired by clinical data (MIMIC-III/IV).  
It is intended for research and educational purposes only and should not be used for real clinical decision-making.

---

## 📌 Future Improvements

- Incorporate real-world EHR datasets
- Improve LLM calibration
- Advanced fairness-aware modeling
- Multimodal integration (labs, imaging)

---

## 👩‍🔬 Authors

- Shalini Majumder, MSc Bioinformatics FT, University of Birmingham
- Ekarsi Lodh, MSc Bioinformatics FT, University of Birmingham 
- Abhijith Sajeev, MSc Bioinformatics FT, University of Birmingham

---

## 📜 License
This project is for academic and research purposes only.
