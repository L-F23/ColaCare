from multiagents_collaboration import process_ehr_patients

my_config = {
    "llm_name": "gpt-4",
    "doctor_num": 3,
    "ehr_dataset_name": "mimic-iv",
    # ... 其他必要字段
}
results = process_ehr_patients(config=my_config, save_root="./my_results", resume=True)