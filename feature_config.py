# config.py
# 实验室指标分类列表（用于智能医疗诊断系统）

# 血常规 (Complete Blood Count)
CBC_PANEL = [
    "Hemoglobin_value",
    "Hematocrit_value",
    "MCV_value",
    "MCH_value",
    "MCHC_value",
    "RDW_value",
    "Platelet Count_value",
    "WBC Count_value",
    "Neutrophils_value",
    "Lymphocytes_value",
    "Monocytes_value",
    "Eosinophils_value",
    "Basophils_value",
    "RBC_value",
    "Absolute Neutrophil Count_value",
    "Absolute Lymphocyte Count_value",
    "Absolute Monocyte Count_value",
    "Absolute Eosinophil Count_value",
    "Absolute Basophil Count_value"
]

# 基础代谢/电解质 (Basic Metabolic Panel)
BMP_PANEL = [
    "Glucose_value",
    "Creatinine_value",
    "Urea Nitrogen_value",
    "Sodium_value",
    "Potassium_value",
    "Chloride_value",
    "Bicarbonate_value",
    "Calcium, Total_value",
    "Anion Gap_value"
]

# 肝功能 (Liver Function Tests)
LFT_PANEL = [
    "Alanine Aminotransferase (ALT)_value",
    "Asparate Aminotransferase (AST)_value",
    "Alkaline Phosphatase_value",
    "Gamma Glutamyltransferase_value",
    "Bilirubin, Total_value",
    "Bilirubin, Direct_value",
    "Bilirubin, Indirect_value",
    "Albumin_value",
    "Total Protein, Ascites_value",
    "Total Protein, Body Fluid_value",
    "Total Protein, CSF_value",
    "Total Protein, Joint Fluid_value",
    "Total Protein, Pleural_value",
    "Total Protein, Urine_value",
]

# 肾功能相关 (Renal Function)
RENAL_PANEL = [
    "Creatinine_value",
    "Urea Nitrogen_value",
    "eAG_value",  # 估算肾小球滤过率
    "Creatinine Clearance_value"
]

# 心肌标志物 (Cardiac Markers)
CARDIAC_PANEL = [
    "Troponin T_value",
    "Creatine Kinase (CK)_value",
    "CK-MB Index_value",
    "NTproBNP_value",
]

# 凝血功能 (Coagulation)
COAGULATION_PANEL = [
    "PT_value",
    "PTT_value",
    "INR(PT)_value",
    "D-Dimer_value",
    "Fibrinogen, Functional_value"
]

# 炎症标志物 (Inflammatory Markers)
INFLAMMATORY_PANEL = [
    "C-Reactive Protein_value",
    "Sedimentation Rate_value",
    "Ferritin_value",
]

# 脂质代谢 (Lipid Panel)
LIPID_PANEL = [
    "Cholesterol, Total_value",
    "Triglycerides_value",
    "Cholesterol, HDL_value",
    "Cholesterol, LDL, Calculated_value",
    "Cholesterol Ratio (Total/HDL)_value"
]

# 甲状腺功能 (Thyroid Panel)
THYROID_PANEL = [
    "Thyroid Stimulating Hormone_value",
    "Thyroxine (T4)_value",
    "Thyroxine (T4), Free_value",
    "Triiodothyronine (T3)_value"
]

# 常见肿瘤标志物 (Tumor Markers)
TUMOR_MARKERS = [
    "Carcinoembyronic Antigen (CEA)_value",
    "CA-125_value",
    "CA 19-9_value",
    "Alpha-Fetoprotein_value",
    "Prostate Specific Antigen_value"
]

# 维生素与微量元素 (Vitamins & Minerals)
VITAMINS_MINERALS = [
    "Vitamin B12_value",
    "Folate_value",
    "Ferritin_value",
    "Iron_value",
    "Iron Binding Capacity, Total_value",
    "25-OH Vitamin D_value",
    "Magnesium_value",
    "Phosphate_value"
]

# 感染性疾病血清学 (Infectious Disease Serology)
INFECTIOUS_PANEL = [
    "HIV 1 Viral Load_value",
    "Hepatitis B Viral Load_value",
    "Hepatitis C Viral Load_value",
    "CMV IgG Ab Value_value",
    "Epstein-Barr Virus IgG Ab Value_value",
    "Epstein-Barr Virus IgM Ab Value_value",
    "H. pylori IgG Ab Value_value"
]

# 全血其他常见指标 (Other Common Blood Tests)
OTHER_COMMON = [
    "Lactate_value",
    "Ammonia_value",
    "Lipase_value",
    "Amylase_value",
    "Uric Acid_value",
    "Homocysteine_value",
    "Cortisol_value",
    "Parathyroid Hormone_value"
]

# 综合所有上述指标（用于一键调用）
ALL_LAB_FEATURES = (
    CBC_PANEL + BMP_PANEL + LFT_PANEL + RENAL_PANEL +
    CARDIAC_PANEL + COAGULATION_PANEL + INFLAMMATORY_PANEL +
    LIPID_PANEL + THYROID_PANEL + TUMOR_MARKERS +
    VITAMINS_MINERALS + INFECTIOUS_PANEL + OTHER_COMMON
)

# 去重（如有重复）
ALL_LAB_FEATURES = list(dict.fromkeys(ALL_LAB_FEATURES))