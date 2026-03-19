import os
import random
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from ehr_datasets.utils.tools import forward_fill_pipeline, normalize_dataframe, export_missing_mask, export_record_time, export_note

from feature_config import ALL_LAB_FEATURES

processed_data_dir = os.path.join("./ehr_datasets/mimic-iv", 'processed')
os.makedirs(processed_data_dir, exist_ok=True)
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# ==================== 根据实际字段定义特征 ====================
# 原始数据中的可用字段
all_columns = [
    'subject_id', 'gender', 'anchor_age', 'anchor_year', 'anchor_year_group', 'dod',
    'hadm_id', 'admittime', 'dischtime', 'deathtime', 'admission_type', 'admit_provider_id',
    'admission_location', 'discharge_location', 'insurance', 'language', 'marital_status',
    'race', 'edregtime', 'edouttime', 'hospital_expire_flag', 'stay_id', 'first_careunit',
    'last_careunit', 'intime', 'outtime', 'los', 'diagnoses', 'procedures', 'prescriptions'
]  

# Record feature names
basic_records = ['RecordID', 'PatientID', 'RecordTime']
target_features = ['Outcome', 'LOS', 'Readmission', 'diagnoses', 'procedures']
icd_codes = ['diag_icd_code', 'proc_icd_code', 'pres_icd_code']
demographic_features = ['Sex', 'Age']
# labtest_features = ['Capillary refill rate', 'Glascow coma scale eye opening', 'Glascow coma scale motor response', 'Glascow coma scale total', 'Glascow coma scale verbal response', 'Diastolic blood pressure', 'Fraction inspired oxygen', 'Glucose', 'Heart Rate', 'Height', 'Mean blood pressure', 'Oxygen saturation', 'Respiratory rate', 'Systolic blood pressure', 'Temperature', 'Weight', 'pH']
# categorical_labtest_features = ['Capillary refill rate', 'Glascow coma scale eye opening', 'Glascow coma scale motor response', 'Glascow coma scale total', 'Glascow coma scale verbal response']
# numerical_labtest_features = ['Diastolic blood pressure', 'Fraction inspired oxygen', 'Glucose', 'Heart Rate', 'Height', 'Mean blood pressure', 'Oxygen saturation', 'Respiratory rate', 'Systolic blood pressure', 'Temperature', 'Weight', 'pH']
labtest_features = ['Glucose', 'Oxygen saturation', 'Temperature','pH'] 
categorical_labtest_features = None
numerical_labtest_features = ['Glucose', 'Oxygen saturation', 'Temperature', 'pH']

normalize_features = ['Age'] + numerical_labtest_features + ['LOS']

# 提取所有以 '_value' 结尾的列作为实验室检测特征
# labtest_features = [col for col in all_columns if col.endswith('_value')]
# 假设所有 labtest_features 均为数值型（无明确分类特征）
categorical_labtest_features = []          # 原代码中的分类特征不存在
numerical_labtest_features = labtest_features

# 需要归一化的特征：年龄 + 所有数值型实验室特征 + LOS
normalize_features = ['Age'] + numerical_labtest_features + ['LOS']

# ==================== 读取数据并创建所需列 ====================
print("=== begin to read data from .parquet ===")
df = pd.read_parquet(os.path.join(processed_data_dir, 'mimic4_formatted_ehr.parquet'))
print("=== successfully read data ===")

# for col in all_columns:
#     print(f"{col} contains: {df[col].iloc[0]}")

# ============ process some basic items =========== #
# ============ in case too many dimension 'str' ==== #
# print("============ processing some basic items ===========")
# df['gender'] = df['gender'].map({'M': 0, 'F': 1}).fillna(-1).astype(int)

# print("============ processed some basic items ===========")
# ============ process some basic items =========== #

# 映射字段
df['PatientID'] = df['subject_id']
df['AdmissionID'] = df['hadm_id']
df['RecordTime'] = df['intime']                # 使用 ICU 入科时间作为记录时间
df['Sex'] = df['gender']                       # 性别
df['Age'] = df['anchor_age']                   # 基线年龄
df['Outcome'] = df['hospital_expire_flag']     
df['LOS'] = df['los']                          # ICU 住院时长
df['Glucose'] = df['Glucose_value']            # 血糖
df['pH'] = df['pH_value']
df['Oxygen saturation'] = df['Oxygen Saturation_value'] 
df['Temperature'] = df['Temperature_value']
print("successfully reflect")

# generate RecordID
if "RecordID" not in df.columns:
    df["RecordID"] = df["PatientID"].astype(str) + "_" + df["AdmissionID"].astype(str)
    print("create Record ID")

# Judge if readmision happened
df['admittime'] = pd.to_datetime(df['admittime'])
df['dischtime'] = pd.to_datetime(df['dischtime'])
df_sort_admi = df.sort_values(['subject_id', 'admittime']).reset_index(drop=True)
df_sort_admi['next_admission'] = df_sort_admi.groupby('subject_id')['admittime'].shift(-1)
df_sort_admi['readmission_days'] = (df_sort_admi['next_admission'] - df_sort_admi['dischtime']).dt.total_seconds()/(24*3600)
df_sort_admi['Readmission'] = ((df_sort_admi['next_admission'].notna()) & 
                               (df_sort_admi['readmission_days'] <= 30) &
                               (df_sort_admi['hospital_expire_flag']==0)).astype(int)
print("successfully sort admission time")
df = df.merge(df_sort_admi[['hadm_id', 'Readmission']], on='hadm_id', how='left')
print("successfully generate Readmission")

# 仅保留所需列（基本记录、目标、人口统计学、实验室特征）
keep_cols = basic_records + target_features + demographic_features + labtest_features + ALL_LAB_FEATURES
df = df[keep_cols]
print("successfully keep columns")

# 确保数据按 RecordID 和 RecordTime 排序（若为静态数据，则每个 RecordID 仅一行）
df = df.sort_values(by=['RecordID', 'RecordTime']).reset_index(drop=True)
print("sorted by RecordID and RecordTime")


df.to_csv("mimic-iv.csv", index=False, encoding='utf-8-sig')
print("successfully store .csv")
# ==================== 数据集划分 ====================
# Group the dataframe by `RecordID`
grouped = df.groupby('RecordID')

# Get the patient IDs and outcomes
patients = np.array(list(grouped.groups.keys()))
patients_outcome = np.array([grouped.get_group(patient_id)['Outcome'].iloc[0] for patient_id in patients])

# unique, counts = np.unique(patients_outcome, return_counts=True)
# with open("uc_pairs.txt", "w") as f:
#     f.write(str(dict(zip(unique, counts))))
# print("pairs save to txt")

# Get the train_val/test patient IDs
train_val_patients, test_patients = train_test_split(patients, test_size=1/100, random_state=SEED, stratify=patients_outcome)

# Get the train/val patient IDs
train_val_patients_outcome = np.array([grouped.get_group(patient_id)['Outcome'].iloc[0] for patient_id in train_val_patients])
train_patients, val_patients = train_test_split(train_val_patients, test_size=4/99, random_state=SEED, stratify=train_val_patients_outcome)
test_df = df[df['RecordID'].isin(test_patients)]
print("successfully group by RecordID")

# ==================== LLM 设置：导出缺失掩码、记录时间、原始数据 ====================
print("start to export missing mask")
train_missing_mask = export_missing_mask(df, demographic_features, labtest_features, id_column='RecordID')
val_missing_mask = export_missing_mask(df, demographic_features, labtest_features, id_column='RecordID')
test_missing_mask = export_missing_mask(test_df, demographic_features, labtest_features, id_column='RecordID')

print("start to export record time")
train_record_time = export_record_time(df, id_column='RecordID')
val_record_time = export_record_time(df, id_column='RecordID')
test_record_time = export_record_time(test_df, id_column='RecordID')

print("start to forward fill")
_, train_raw_x, _, _ = forward_fill_pipeline(df, None, demographic_features, labtest_features, target_features, [], id_column='RecordID')
_, val_raw_x, _, _ = forward_fill_pipeline(df, None, demographic_features, labtest_features, target_features, [], id_column='RecordID')
_, test_raw_x, _, _ = forward_fill_pipeline(test_df, None, demographic_features, labtest_features, target_features, [], id_column='RecordID')

save_dir = os.path.join(processed_data_dir, 'processed', 'fold_1')
os.makedirs(save_dir, exist_ok=True)

print("start storing missing data to pickle")
pd.to_pickle(train_missing_mask, os.path.join(save_dir, 'train_missing_mask.pkl'))
pd.to_pickle(val_missing_mask, os.path.join(save_dir, 'val_missing_mask.pkl'))
pd.to_pickle(test_missing_mask, os.path.join(save_dir, 'test_missing_mask.pkl'))
pd.to_pickle(train_record_time, os.path.join(save_dir, 'train_record_time.pkl'))
pd.to_pickle(val_record_time, os.path.join(save_dir, 'val_record_time.pkl'))
pd.to_pickle(test_record_time, os.path.join(save_dir, 'test_record_time.pkl'))
pd.to_pickle(train_raw_x, os.path.join(save_dir, 'train_raw_x.pkl'))
pd.to_pickle(val_raw_x, os.path.join(save_dir, 'val_raw_x.pkl'))
pd.to_pickle(test_raw_x, os.path.join(save_dir, 'test_raw_x.pkl'))
print("successfully store missing data to pickle")

# ==================== ML/DL 设置：处理分类特征（若无则跳过 one-hot）====================
if categorical_labtest_features:
    one_hot = pd.get_dummies(df[categorical_labtest_features], columns=categorical_labtest_features, prefix_sep='->', dtype=float)
    columns = df.columns.to_list()
    column_start_idx = columns.index(categorical_labtest_features[0])
    column_end_idx = columns.index(categorical_labtest_features[-1])
    df = pd.concat([df.loc[:, columns[:column_start_idx]], one_hot, df.loc[:, columns[column_end_idx + 1:]]], axis=1)
    ehr_categorical_labtest_features = one_hot.columns.to_list()
    ehr_labtest_features = ehr_categorical_labtest_features + numerical_labtest_features
else:
    ehr_labtest_features = numerical_labtest_features   # 无分类特征时直接使用数值特征

require_impute_features = ehr_labtest_features

# ==================== 再次分组确认无数据泄露 ====================
print("Train patients size:", len(train_patients))
print("Validation patients size:", len(val_patients))
print("Test patients size:", len(test_patients))

assert len(set(train_patients) & set(val_patients)) == 0
assert len(set(train_patients) & set(test_patients)) == 0
assert len(set(val_patients) & set(test_patients)) == 0

# ==================== 创建训练/验证/测试 DataFrame ====================
train_df = df[df['RecordID'].isin(train_patients)]
val_df = df[df['RecordID'].isin(val_patients)]
test_df = df[df['RecordID'].isin(test_patients)]

# 归一化：基于训练集 5%~95% 分位数范围计算均值和标准差
train_df, val_df, test_df, default_fill, los_info, train_mean, train_std = normalize_dataframe(
    train_df, val_df, test_df, normalize_features, id_column="RecordID"
)

# 前向填充（若数据为静态，则每组仅一行，填充无影响）
train_df, train_x, train_y, train_pid = forward_fill_pipeline(
    train_df, default_fill, demographic_features, ehr_labtest_features, target_features, require_impute_features, id_column="RecordID"
)
val_df, val_x, val_y, val_pid = forward_fill_pipeline(
    val_df, default_fill, demographic_features, ehr_labtest_features, target_features, require_impute_features, id_column="RecordID"
)
test_df, test_x, test_y, test_pid = forward_fill_pipeline(
    test_df, default_fill, demographic_features, ehr_labtest_features, target_features, require_impute_features, id_column="RecordID"
)

# 保存处理后的数据
pd.to_pickle(train_x, os.path.join(save_dir, "train_x.pkl"))
pd.to_pickle(train_y, os.path.join(save_dir, "train_y.pkl"))
pd.to_pickle(train_pid, os.path.join(save_dir, "train_pid.pkl"))
pd.to_pickle(val_x, os.path.join(save_dir, "val_x.pkl"))
pd.to_pickle(val_y, os.path.join(save_dir, "val_y.pkl"))
pd.to_pickle(val_pid, os.path.join(save_dir, "val_pid.pkl"))
pd.to_pickle(test_x, os.path.join(save_dir, "test_x.pkl"))
pd.to_pickle(test_y, os.path.join(save_dir, "test_y.pkl"))
pd.to_pickle(test_pid, os.path.join(save_dir, "test_pid.pkl"))
pd.to_pickle(los_info, os.path.join(save_dir, "los_info.pkl"))

# 保存实验室特征列表（数值型）
pd.to_pickle(numerical_labtest_features, os.path.join(save_dir, 'labtest_features.pkl'))

# 保存统计信息（按结局分组）
pd.to_pickle(df.groupby('Outcome').get_group(0).describe().to_dict('dict'), os.path.join(save_dir, 'survival.pkl'))
pd.to_pickle(df.groupby('Outcome').get_group(1).describe().to_dict('dict'), os.path.join(save_dir, 'dead.pkl'))
pd.to_pickle(df[['PatientID', 'Sex', 'Age']].groupby('PatientID').first().to_dict('index'), os.path.join(save_dir, 'basic.pkl'))