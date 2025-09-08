# chronic_hackathon_single_patient.py
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, balanced_accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.calibration import calibration_curve
import joblib
import warnings
warnings.filterwarnings('ignore')

# -----------------------------
# 0. Paths (EDIT THESE)
# ----------------------------
# 
# 
mimic_root = "C:/Users/suhan/OneDrive/Desktop/PYTHON/AIPROJ/mimic-iv-clinical-database-demo-2.2/"
mimic_hosp = os.path.join(mimic_root, "hosp")
mimic_icu  = os.path.join(mimic_root, "icu")

eicu_root = "C:/Users/suhan/OneDrive/Desktop/PYTHON/AIPROJ/eicu-collaborative-research-database-demo-2.0.1/"

CHUNK = 200000   # adjust if memory constrained

CHUNK = 200000   # adjust if memory constrained

# -----------------------------
# 1. Target labels & thresholds
# -----------------------------
DESIRED_VITAL_LABELS = ['Heart Rate','Respiratory Rate','SpO2','Temperature','Non Invasive BP systolic','Non Invasive BP diastolic']
DESIRED_LAB_LABELS   = ['GLUCOSE','SODIUM','POTASSIUM','CREATININE','HEMOGLOBIN']
ALL_MEASURES = DESIRED_VITAL_LABELS + DESIRED_LAB_LABELS

def compute_critical_score_from_df(df):
    s = 0
    if 'GLUCOSE' in df.columns: s += ((df['GLUCOSE'] < 80) | (df['GLUCOSE'] > 130)).sum()
    if 'SODIUM' in df.columns: s += ((df['SODIUM'] < 136) | (df['SODIUM'] > 144)).sum()
    if 'POTASSIUM' in df.columns: s += ((df['POTASSIUM'] < 3.6) | (df['POTASSIUM'] > 4.8)).sum()
    if 'CREATININE' in df.columns: s += (df['CREATININE'] > 1.2).sum()
    if 'HEMOGLOBIN' in df.columns: s += (df['HEMOGLOBIN'] < 12.5).sum()
    if 'Heart Rate' in df.columns: s += ((df['Heart Rate'] < 60) | (df['Heart Rate'] > 95)).sum()
    if 'Respiratory Rate' in df.columns: s += ((df['Respiratory Rate'] < 12) | (df['Respiratory Rate'] > 20)).sum()
    if 'SpO2' in df.columns: s += (df['SpO2'] < 95).sum()
    if 'Temperature' in df.columns: s += ((df['Temperature'] < 36) | (df['Temperature'] > 37.2)).sum()
    if 'Non Invasive BP systolic' in df.columns: s += ((df['Non Invasive BP systolic'] < 95) | (df['Non Invasive BP systolic'] > 135)).sum()
    if 'Non Invasive BP diastolic' in df.columns: s += ((df['Non Invasive BP diastolic'] < 60) | (df['Non Invasive BP diastolic'] > 85)).sum()
    return int(s)

def assign_deterioration_level(score):
    if score == 0: return "No deterioration"
    elif score < 5: return "Minor"
    elif score < 10: return "Moderate"
    else: return "Severe"

def safe_to_datetime(series): return pd.to_datetime(series, errors='coerce')

# -----------------------------
# 2. Load MIMIC mapping (d_items / d_labitems)
# -----------------------------
d_items_path = os.path.join(mimic_icu, "d_items.csv.gz")
d_labitems_path = os.path.join(mimic_hosp, "d_labitems.csv.gz")
d_items = None
d_labitems = None
if os.path.exists(d_items_path):
    d_items = pd.read_csv(d_items_path, compression='gzip', usecols=['itemid','label'])
if os.path.exists(d_labitems_path):
    d_labitems = pd.read_csv(d_labitems_path, compression='gzip', usecols=['itemid','label'])

# -----------------------------
# 3. Load admissions/patient tables for mapping and death-time (optional)
# -----------------------------
mimic_adm_path = os.path.join(mimic_hosp, "admissions.csv.gz")
mimic_pat_path = os.path.join(mimic_hosp, "patients.csv.gz")
mimic_adm = pd.read_csv(mimic_adm_path, compression='gzip', usecols=['subject_id','hadm_id','admittime','dischtime'])
mimic_adm['admittime'] = safe_to_datetime(mimic_adm['admittime'])
mimic_adm['dischtime'] = safe_to_datetime(mimic_adm['dischtime'])
mimic_pat = None
if os.path.exists(mimic_pat_path):
    mimic_pat = pd.read_csv(
        mimic_pat_path,
        compression='gzip'
    )
    # Only keep the columns that actually exist in the demo
    keep_cols = ['subject_id','gender','anchor_age']
    mimic_pat = mimic_pat[[c for c in keep_cols if c in mimic_pat.columns]].copy()
    
    # If full dataset is used (not demo), then convert death dates safely
    if 'deathtime' in mimic_pat.columns:
        mimic_pat['deathtime'] = safe_to_datetime(mimic_pat['deathtime'])
    if 'dod' in mimic_pat.columns:
        mimic_pat['dod'] = safe_to_datetime(mimic_pat['dod'])

# eICU patient
eicu_patient_path = os.path.join(eicu_root, "patient.csv.gz")
eicu_patient = None
if os.path.exists(eicu_patient_path):
    eicu_patient = pd.read_csv(eicu_patient_path, compression='gzip')
    # ensure patient_id column exists
    if 'patientunitstayid' in eicu_patient.columns:
        eicu_patient = eicu_patient.rename(columns={'patientunitstayid':'patient_id'})

# -----------------------------
# 4. Extract MIMIC patient-day timeseries (vitals + labs)
# -----------------------------
print("Extracting MIMIC time-series (chunked). This may take time...")

mimic_rows = []  # holds small frames with hadm_id, day_index, measure, value

# chartevents -> vitals
chartevents_path = os.path.join(mimic_icu, "chartevents.csv.gz")
if os.path.exists(chartevents_path) and d_items is not None:
    usecols = ['subject_id','hadm_id','itemid','charttime','valuenum']
    for chunk in pd.read_csv(chartevents_path, compression='gzip', usecols=usecols, chunksize=CHUNK):
        chunk = chunk.dropna(subset=['valuenum'])
        chunk = chunk.merge(d_items, on='itemid', how='left')
        chunk = chunk[chunk['label'].isin(DESIRED_VITAL_LABELS)].copy()
        if chunk.empty:
            continue
        chunk['charttime'] = safe_to_datetime(chunk['charttime'])
        chunk = chunk.merge(mimic_adm[['hadm_id','admittime']], on='hadm_id', how='left')
        chunk = chunk.dropna(subset=['admittime','charttime'])
        chunk['day_index'] = ((chunk['charttime'] - chunk['admittime']).dt.total_seconds() // (3600*24)).astype(int)
        chunk = chunk[chunk['day_index'] >= 0]
        tmp = chunk[['hadm_id','day_index','label','valuenum']].rename(columns={'label':'measure','valuenum':'value'})
        mimic_rows.append(tmp)

# labevents -> labs
labevents_path = os.path.join(mimic_hosp, "labevents.csv.gz")
if os.path.exists(labevents_path) and d_labitems is not None:
    usecols = ['subject_id','hadm_id','itemid','charttime','valuenum']
    for chunk in pd.read_csv(labevents_path, compression='gzip', usecols=usecols, chunksize=CHUNK):
        chunk = chunk.dropna(subset=['valuenum'])
        chunk = chunk.merge(d_labitems, on='itemid', how='left')
        chunk = chunk[chunk['label'].isin(DESIRED_LAB_LABELS)].copy()
        if chunk.empty:
            continue
        chunk['charttime'] = safe_to_datetime(chunk['charttime'])
        chunk = chunk.merge(mimic_adm[['hadm_id','admittime']], on='hadm_id', how='left')
        chunk = chunk.dropna(subset=['admittime','charttime'])
        chunk['day_index'] = ((chunk['charttime'] - chunk['admittime']).dt.total_seconds() // (3600*24)).astype(int)
        chunk = chunk[chunk['day_index'] >= 0]
        tmp = chunk[['hadm_id','day_index','label','valuenum']].rename(columns={'label':'measure','valuenum':'value'})
        mimic_rows.append(tmp)

if mimic_rows:
    mimic_ts = pd.concat(mimic_rows, ignore_index=True)
    mimic_wide = mimic_ts.groupby(['hadm_id','day_index','measure'])['value'].mean().reset_index()
    mimic_wide = mimic_wide.pivot_table(index=['hadm_id','day_index'], columns='measure', values='value').reset_index()
    # map hadm_id -> subject_id using admissions table (so we can stitch across admissions)
    mimic_wide = mimic_wide.merge(mimic_adm[['hadm_id','subject_id','admittime']], on='hadm_id', how='left')
    # reorder and rename
    if 'subject_id' in mimic_wide.columns:
        mimic_wide = mimic_wide.rename(columns={'subject_id':'patient_base_id'})
    else:
        mimic_wide['patient_base_id'] = mimic_wide['hadm_id'].astype(str).apply(lambda x: "MIMIC_"+x)
    # ensure columns exist
    for col in ALL_MEASURES:
        if col not in mimic_wide.columns:
            mimic_wide[col] = np.nan
    mimic_wide['source'] = 'MIMIC'
    # create a per-row absolute day index within stitched timeline later
else:
    mimic_wide = pd.DataFrame(columns=['hadm_id','day_index','patient_base_id'] + ALL_MEASURES + ['admittime','source'])
    print("No MIMIC timeseries extracted.")

print("MIMIC patient-day rows:", len(mimic_wide))

# -----------------------------
# 5. Extract eICU patient-day timeseries (vitalPeriodic + lab)
# -----------------------------
print("Extracting eICU time-series...")

eicu_rows = []
vital_periodic_path = os.path.join(eicu_root, "vitalPeriodic.csv.gz")
if os.path.exists(vital_periodic_path):
    # read full (demo likely small), else you can chunk similarly
    vp = pd.read_csv(vital_periodic_path, compression='gzip')
    # map columns to our labels
    rename_map = {}
    for col in vp.columns:
        lc = col.lower()
        if 'heartrate' in lc or lc == 'heartrate':
            rename_map[col] = 'Heart Rate'
        if 'respiratory' in lc or 'respiratoryrate' in lc:
            rename_map[col] = 'Respiratory Rate'
        if 'spo2' in lc:
            rename_map[col] = 'SpO2'
        if 'temp' in lc or 'temperature' in lc:
            rename_map[col] = 'Temperature'
        if 'systolic' in lc:
            rename_map[col] = 'Non Invasive BP systolic'
        if 'diastolic' in lc:
            rename_map[col] = 'Non Invasive BP diastolic'
    vp = vp.rename(columns=rename_map)
    # identify id & time columns
    pid_col = None
    for cand in ['patientunitstayid','patient_id','patientid']:
        if cand in vp.columns:
            pid_col = cand
            break
    time_col = None
    for cand in ['observationoffset','observationoffsetmins','observationoffsetmins','offset','charttime','observationtime','datetimerecorded']:
        if cand in vp.columns:
            time_col = cand
            break
    if pid_col is not None:
        # compute day_index
        if time_col is not None:
            vp['day_index'] = (vp[time_col].astype(float) // (60*24)).astype(int)
        else:
            vp['row_idx'] = vp.groupby(pid_col).cumcount()
            vp['day_index'] = (vp['row_idx'] // 1).astype(int)
        vp = vp.rename(columns={pid_col:'patient_id'})
        # melt measures
        value_cols = [c for c in rename_map.values() if c in vp.columns]
        if value_cols:
            vp_melt = vp.melt(id_vars=['patient_id','day_index'], value_vars=value_cols, var_name='measure', value_name='value')
            vp_melt = vp_melt.dropna(subset=['value'])
            eicu_rows.append(vp_melt)

# eICU labs
lab_path = os.path.join(eicu_root, "lab.csv.gz")
if os.path.exists(lab_path):
    lab = pd.read_csv(lab_path, compression='gzip')
    if 'labname' in lab.columns:
        lab['labname_l'] = lab['labname'].str.lower()
        lab_map = {'glucose':'GLUCOSE','sodium':'SODIUM','potassium':'POTASSIUM','creatinine':'CREATININE','hemoglobin':'HEMOGLOBIN'}
        lab = lab[lab['labname_l'].isin(lab_map.keys())].copy()
        lab['measure'] = lab['labname_l'].map(lab_map)
        pid_col = 'patientunitstayid' if 'patientunitstayid' in lab.columns else ('patient_id' if 'patient_id' in lab.columns else None)
        if pid_col is not None:
            lab = lab.rename(columns={pid_col:'patient_id'})
            if 'labresultoffset' in lab.columns:
                lab['day_index'] = (lab['labresultoffset'] // (60*24)).astype(int)
            else:
                lab['day_index'] = lab.groupby('patient_id').cumcount()
            lab_small = lab[['patient_id','day_index','measure','labresult']].rename(columns={'labresult':'value'})
            eicu_rows.append(lab_small)

if eicu_rows:
    eicu_ts = pd.concat(eicu_rows, ignore_index=True)
    eicu_wide = eicu_ts.groupby(['patient_id','day_index','measure'])['value'].mean().reset_index()
    eicu_wide = eicu_wide.pivot_table(index=['patient_id','day_index'], columns='measure', values='value').reset_index()
    # ensure expected cols
    for col in ALL_MEASURES:
        if col not in eicu_wide.columns:
            eicu_wide[col] = np.nan
    eicu_wide['source'] = 'eICU'
    # rename patient id to patient_base_id for stitching (prefix to avoid collision)
    eicu_wide['patient_base_id'] = eicu_wide['patient_id'].astype(str).apply(lambda x: "EICU_"+x)
else:
    eicu_wide = pd.DataFrame(columns=['patient_id','day_index'] + ALL_MEASURES + ['source','patient_base_id'])
    print("No eICU timeseries extracted.")

print("eICU patient-day rows:", len(eicu_wide))

# -----------------------------
# 6. Prepare stitch-ready patient-day timelines
# -----------------------------
# For MIMIC: we want patient_base_id that groups multiple hadm_id admissions
if not mimic_wide.empty:
    # ensure patient_base_id for MIMIC (prefix)
    mimic_wide['patient_base_id'] = mimic_wide['patient_base_id'].astype(str).apply(lambda x: "MIMIC_"+str(x))
    # select needed cols
    mimic_sel = mimic_wide[['patient_base_id','hadm_id','day_index'] + ALL_MEASURES + ['admittime','source']].copy()
else:
    mimic_sel = pd.DataFrame(columns=['patient_base_id','hadm_id','day_index'] + ALL_MEASURES + ['admittime','source'])

if not eicu_wide.empty:
    eicu_sel = eicu_wide[['patient_base_id','day_index'] + ALL_MEASURES + ['source']].copy()
else:
    eicu_sel = pd.DataFrame(columns=['patient_base_id','day_index'] + ALL_MEASURES + ['source'])

# combine them
combined_pd = pd.concat([mimic_sel, eicu_sel], axis=0, ignore_index=True, sort=False)
combined_pd = combined_pd.fillna(np.nan)
print("Combined patient-day rows (before stitching):", len(combined_pd))

# -----------------------------
# 7. Stitch multiple stays for each patient_base_id into a continuous timeline
# -----------------------------
print("Stitching multiple admissions per patient to create long timelines...")

stitched_rows = []
for pid, g in combined_pd.groupby('patient_base_id'):
    # sort by admittime if present (MIMIC rows have admittime), else by availability of day_index then source
    g = g.sort_values(by=['admittime','day_index'], na_position='last') if 'admittime' in g.columns else g.sort_values(by=['day_index'], na_position='last')
    # We will compute an accumulated_offset to add to the day_index of subsequent stays
    accumulated_offset = 0
    # We treat each block of contiguous hadm_id/day_index as one stay — for eICU rows hadm_id will be NaN
    # We'll process unique hadm_id groups if present, else treat whole g as single timeline
    if 'hadm_id' in g.columns and g['hadm_id'].notna().any():
        # iterate by hadm_id groups (NaN hadm_id grouped together)
        for hid, sub in g.groupby('hadm_id', dropna=False):
            sub = sub.sort_values('day_index')
            if sub.empty:
                continue
            # compute new_day_index = accumulated_offset + original day_index
            sub2 = sub.copy()
            sub2['stitched_day'] = sub2['day_index'].astype(float).fillna(0).astype(int) + accumulated_offset
            # update accumulated_offset for next block
            max_day = sub2['stitched_day'].max()
            accumulated_offset = max_day + 1  # leave a 1-day gap
            stitched_rows.append(sub2)
    else:
        # no hadm_id (likely eICU only), just take entries sorted and assign stitched_day as running index
        sub = g.sort_values('day_index')
        if sub.empty:
            continue
        sub2 = sub.copy()
        # if day_index starts at 0 within stay, use it; but to make continuous, we reindex by order
        base = 0
        sub2['stitched_day'] = sub2['day_index'].astype(float).fillna(0).astype(int) + accumulated_offset
        # update accumulated_offset
        max_day = sub2['stitched_day'].max()
        accumulated_offset = max_day + 1
        stitched_rows.append(sub2)

if stitched_rows:
    stitched = pd.concat(stitched_rows, ignore_index=True, sort=False)
else:
    stitched = pd.DataFrame(columns=combined_pd.columns.tolist()+['stitched_day'])
print("Stitched patient-day rows:", len(stitched))

# -----------------------------
# 8. Build sliding windows: past 30-120 days -> next 90 days
# -----------------------------
PAST_DAYS = 10   # base window length (we require at least PAST_MIN days)
PAST_MIN = 5
PAST_MAX = 15
FUTURE_DAYS = 5

print("Building sliding windows and labels...")
samples = []
for pid, g in tqdm(stitched.groupby('patient_base_id'), desc="Patients"):
    sub = g.sort_values('stitched_day')
    days = sub['stitched_day'].dropna().unique()
    if len(days) == 0:
        continue
    # slide start across realistic range: use start at min day up to max - (past+future)
    min_day = int(days.min())
    max_day = int(days.max())
    # For efficiency, step by 1 day
    for start in range(min_day, max_day - (PAST_DAYS + FUTURE_DAYS) + 1):
        past_start = start
        past_end = past_start + PAST_DAYS
        future_start = past_end
        future_end = future_start + FUTURE_DAYS

        past_df = sub[(sub['stitched_day'] >= past_start) & (sub['stitched_day'] < past_end)]
        future_df = sub[(sub['stitched_day'] >= future_start) & (sub['stitched_day'] < future_end)]

        if len(past_df) < PAST_MIN or len(past_df) > PAST_MAX or len(future_df) == 0:
            continue

        # aggregate past features
        feat = {}
        for col in ALL_MEASURES:
            if col in past_df.columns:
                vals = past_df[col].dropna().values
                feat[f"{col}_mean"] = float(np.nanmean(vals)) if len(vals)>0 else np.nan
                feat[f"{col}_min"] = float(np.nanmin(vals)) if len(vals)>0 else np.nan
                feat[f"{col}_max"] = float(np.nanmax(vals)) if len(vals)>0 else np.nan
                # slope
                try:
                    sorted_vals = past_df.sort_values('stitched_day')[col].fillna(method='ffill').values
                    if len(sorted_vals) >= 2:
                        feat[f"{col}_slope"] = float((sorted_vals[-1] - sorted_vals[0]) / max(1, len(sorted_vals)-1))
                    else:
                        feat[f"{col}_slope"] = 0.0
                except Exception:
                    feat[f"{col}_slope"] = 0.0
            else:
                feat[f"{col}_mean"] = np.nan
                feat[f"{col}_min"] = np.nan
                feat[f"{col}_max"] = np.nan
                feat[f"{col}_slope"] = np.nan

        # compute deterioration label using future_df
        score = compute_critical_score_from_df(future_df)
        det_label = assign_deterioration_level(score)

        sample = {
            'patient_base_id': pid,
            'window_start': past_start,
            'window_end': past_end,
            'label_next90d_deterioration': det_label
        }
        sample.update(feat)
        samples.append(sample)

forecast_df = pd.DataFrame(samples)
print("Forecast dataset size:", forecast_df.shape)
if forecast_df.empty:
    print("No forecast windows found. That would be surprising after stitching — check stitched timelines.")
    raise SystemExit(0)

# -----------------------------
# 9. Train/evaluate model (deterioration label) using XGBoost + SMOTE
# -----------------------------
print("Training enhanced XGBoost pipeline with SMOTE and hyperparameter tuning...")

# Columns for modeling
exclude_cols = ['patient_base_id','window_start','window_end','label_next90d_deterioration']
model_cols = [c for c in forecast_df.columns if c not in exclude_cols]
X = forecast_df[model_cols].copy()
y = forecast_df['label_next90d_deterioration']

# Encode labels
le = LabelEncoder()
y_enc = le.fit_transform(y)

# Split train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y_enc, stratify=y_enc, test_size=0.2, random_state=42
)

# Define pipeline
pipeline = ImbPipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('smote', SMOTE(random_state=42)),
    ('xgb', XGBClassifier(
        eval_metric='mlogloss',
        use_label_encoder=False,
        random_state=42
    ))
])

# Hyperparameter grid
param_grid = {
    'xgb__n_estimators': [200, 300, 400],
    'xgb__max_depth': [4, 6, 8],
    'xgb__learning_rate': [0.01, 0.05, 0.1],
    'xgb__subsample': [0.6, 0.8, 1.0],
    'xgb__colsample_bytree': [0.6, 0.8, 1.0],
    'xgb__gamma': [0, 1, 5],
    'xgb__reg_alpha': [0, 0.01, 0.1],
    'xgb__reg_lambda': [1, 1.5, 2]
}

# Stratified CV
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Randomized search
search = RandomizedSearchCV(
    pipeline,
    param_distributions=param_grid,
    n_iter=20,
    cv=cv,
    scoring='accuracy',
    n_jobs=-1,
    verbose=2,
    random_state=42
)

# Train
search.fit(X_train, y_train)
best_model = search.best_estimator_

# Save the trained model for future use
joblib.dump(best_model, 'chronic_deterioration_model.pkl')
joblib.dump(le, 'label_encoder.pkl')
print("Model saved as 'chronic_deterioration_model.pkl'")

# Predict
y_pred = best_model.predict(X_test)
y_test_labels = le.inverse_transform(y_test)
y_pred_labels = le.inverse_transform(y_pred)

# Evaluation
print("\nClassification Report:")

# 1. Get predicted probabilities from the XGBoost model
y_proba = best_model.predict_proba(X_test)  # shape: (n_samples, n_classes)

# 2. AUROC (one-vs-rest for multiclass)
auroc_scores = {}
for i, cls in enumerate(le.classes_):
    auroc = roc_auc_score((y_test == i).astype(int), y_proba[:, i])
    auroc_scores[cls] = auroc
print("AUROC per class:", auroc_scores)

# 3. AUPRC (one-vs-rest)
auprc_scores = {}
for i, cls in enumerate(le.classes_):
    auprc = average_precision_score((y_test == i).astype(int), y_proba[:, i])
    auprc_scores[cls] = auprc
print("AUPRC per class:", auprc_scores)

# 4. Calibration plot for each class
plt.figure(figsize=(10,7))
for i, cls in enumerate(le.classes_):
    prob_true, prob_pred = calibration_curve((y_test == i).astype(int), y_proba[:, i], n_bins=10)
    plt.plot(prob_pred, prob_true, marker='o', label=f'{cls}')
plt.plot([0,1], [0,1], 'k--')  # perfect calibration
plt.xlabel('Predicted Probability')
plt.ylabel('Observed Frequency')
plt.title('Calibration Curve')
plt.legend()
plt.savefig('calibration_curve.png')
plt.show()

print(classification_report(y_test_labels, y_pred_labels))
print(f"Accuracy: {accuracy_score(y_test_labels, y_pred_labels):.4f}")
print(f"F1-score (macro): {f1_score(y_test_labels, y_pred_labels, average='macro'):.4f}")
print(f"Balanced Accuracy: {balanced_accuracy_score(y_test_labels, y_pred_labels):.4f}")

# Confusion matrix
cm = confusion_matrix(y_test_labels, y_pred_labels, labels=le.classes_)
plt.figure(figsize=(7,5))
sns.heatmap(cm, annot=True, fmt="d", xticklabels=le.classes_, yticklabels=le.classes_, cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix - Deterioration (next 90 days)")
plt.savefig('confusion_matrix.png')
plt.show()

print(f"Best hyperparameters found: {search.best_params_}")
print("Training complete.")

# -----------------------------
# 10. Function to predict for a single patient
# -----------------------------
def predict_single_patient(patient_data, model, label_encoder):
    """
    Predict deterioration risk for a single patient
    
    Parameters:
    patient_data (DataFrame): Patient data with the same structure as training data
    model: Trained model
    label_encoder: Fitted label encoder
    
    Returns:
    dict: Prediction results with probabilities
    """
    # Ensure the data has the same columns as training data
    missing_cols = set(model_cols) - set(patient_data.columns)
    for col in missing_cols:
        patient_data[col] = np.nan
    
    # Reorder columns to match training data
    patient_data = patient_data[model_cols]
    
    # Impute missing values (using the same strategy as training)
    imputer = SimpleImputer(strategy='median')
    patient_data_imputed = imputer.fit_transform(patient_data)
    
    # Make prediction
    prediction_proba = model.predict_proba(patient_data_imputed)
    prediction_class = model.predict(patient_data_imputed)
    
    # Convert back to original labels
    prediction_label = label_encoder.inverse_transform(prediction_class)
    
    # Create result dictionary
    result = {
        'prediction': prediction_label[0],
        'probabilities': dict(zip(label_encoder.classes_, prediction_proba[0]))
    }
    
    return result

# -----------------------------
# 11. Single patient prediction from input file
# -----------------------------
def load_patient_data(file_path):
    """
    Load patient data from CSV file
    File should have the same feature columns as the training data
    """
    try:
        patient_data = pd.read_csv(file_path)
        print(f"Successfully loaded patient data from {file_path}")
        return patient_data
    except Exception as e:
        print(f"Error loading patient data: {e}")
        return None

# # Specify the path to your patient data file
# PATIENT_FILE_PATH = "C:/Users/suhan/OneDrive/Desktop/PYTHON/AIPROJ/deepseek_csv_20250905_8a8113.txt" # EDIT THIS PATH

# # Load the saved model and label encoder
# try:
#     model = joblib.load('chronic_deterioration_model.pkl')
#     le_loaded = joblib.load('label_encoder.pkl')
#     print("\nModel loaded successfully for single patient prediction")
# except:
#     print("\nUsing the just-trained model for prediction")
#     model = best_model
#     le_loaded = le

# # Load patient data and make prediction
# patient_data = load_patient_data(PATIENT_FILE_PATH)

# if patient_data is not None:
#     prediction_result = predict_single_patient(patient_data, model, le_loaded)

#     print("\nSingle Patient Prediction Results:")
#     print(f"Predicted Deterioration Level: {prediction_result['prediction']}")
#     print("Probability Distribution:")
#     for cls, prob in prediction_result['probabilities'].items():
#         print(f"  {cls}: {prob:.4f}")

#     # Create visualization for the prediction
#     plt.figure(figsize=(10, 6))
#     classes = list(prediction_result['probabilities'].keys())
#     probabilities = list(prediction_result['probabilities'].values())

#     bars = plt.bar(classes, probabilities, color=['green', 'yellow', 'orange', 'red'])
#     plt.xlabel('Deterioration Level')
#     plt.ylabel('Probability')
#     plt.title('Predicted Deterioration Risk for Patient')
#     plt.ylim(0, 1)

#     # Add value labels on bars
#     for bar, prob in zip(bars, probabilities):
#         plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
#                  f'{prob:.3f}', ha='center', va='bottom')

#     plt.tight_layout()
#     plt.savefig('single_patient_prediction.png')
#     plt.show()

#     print("\nPrediction visualization saved as 'single_patient_prediction.png'")
# else:
#     print("Could not load patient data. Please check the file path.")