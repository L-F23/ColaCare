import json
import os
import traceback as tb
import time
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

from utils.hparams import mimic_config as default_config  # 默认配置，也可以作为参数传入
from utils.framework import LeaderAgent, DoctorAgent, Collaboration
from utils.retrieve_utils import RetrievalSystem
from utils.runner_utils import *

def process_ehr_patients(config=None, save_root="./response", resume=True):
    """
    Parameters:
        config (dict): 配置字典，若为 None 则使用默认配置。
        save_root (str): 结果保存的根目录。
        resume (bool): 是否跳过已处理的患者（断点续传）。
    Returns:
        dict: 汇总结果字典，包含每个患者的报告、token 消耗、时间等。
    """
    if config is None:
        config = default_config  # 使用默认配置

    # 初始化智能体
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    leader_agent = LeaderAgent(llm_name=config["llm_name"])
    retrieval_system = RetrievalSystem(
        retriever_name=config["retriever_name"],
        corpus_name=config["corpus_name"]
    )
    doctor_agents = [
        DoctorAgent(config, i, retrieval_system)
        for i in range(config["doctor_num"])
    ]

    # 准备数据
    test_pids, test_y = load_data(config)
    preds = load_preds(config)  # 可能用于对比

    # 创建保存目录
    save_dir = os.path.join(
        save_root,
        f"{config['ehr_dataset_name']}_{config['ehr_task']}_{config['mode']}_"
        f"{'_'.join(config['ehr_model_names'])}_{config['llm_name']}_{config['corpus_name']}"
    )
    os.makedirs(save_dir, exist_ok=True)

    result_json = {}
    all_prompt_tokens = 0
    all_completion_tokens = 0
    total_start = time.time()

    for patient_index, patient_id in tqdm(
        enumerate(test_pids),
        total=len(test_pids),
        desc=f"Processing patients in {config['ehr_dataset_name']} dataset {config['ehr_task']} task {config['mode']} mode"
    ):
        start = time.time()
        sub_save_dir = f"{save_dir}/pid{patient_id}"
        if resume and os.path.exists(os.path.join(sub_save_dir, "meta_final_summary.json")):
            # 如果已经处理完成，跳过
            continue

        os.makedirs(sub_save_dir, exist_ok=True)
        print('PatientID:', patient_id)
        analysis = []
        prompt_tokens = 0
        completion_tokens = 0
        basic_context = None

        try:
            # 多医生并行分析
            def process_agent(doctor_agent, patient_index, patient_id, i, sub_save_dir):
                response, basic_context, messages, prompt_token, completion_token = \
                    doctor_agent.analysis(patient_index, patient_id)

                with open(f"{sub_save_dir}/doctor{i + 1}_review.json", "w") as f:
                    json.dump(response, f)
                with open(f"{sub_save_dir}/doctor{i + 1}_review_messages.json", "w") as f:
                    json.dump(messages, f)
                with open(f"{sub_save_dir}/doctor{i + 1}_review_userprompt.txt", "w") as f:
                    f.write(messages[1]["content"] if len(messages) > 1 else "")

                return response, prompt_token, completion_token, basic_context

            with ThreadPoolExecutor() as executor:
                futures = []
                for i, doctor_agent in enumerate(doctor_agents):
                    futures.append(
                        executor.submit(
                            process_agent,
                            doctor_agent,
                            patient_index,
                            patient_id,
                            i,
                            sub_save_dir
                        )
                    )

                for future in futures:
                    response, pt, ct, bc = future.result()
                    analysis.append(response)
                    prompt_tokens += pt
                    completion_tokens += ct
                    basic_context = bc  # 所有医生应返回相同的 basic_context，这里取最后一个

            # Leader 初次总结
            leader_agent.set_basic_info(basic_context)
            summary_content, messages, pt, ct = leader_agent.summary(analysis, is_initial=True)
            json.dump(summary_content, open(f"{sub_save_dir}/meta_summary.json", "w"))
            json.dump(messages, open(f"{sub_save_dir}/meta_messages.json", "w"))
            with open(f"{sub_save_dir}/meta_userprompt.txt", "w") as f:
                f.write(messages[1]["content"])
            prompt_tokens += pt
            completion_tokens += ct

            # 协作讨论
            collaboration = Collaboration(
                leader_agent,
                doctor_agents,
                summary_content["answer"],
                summary_content["report"],
                sub_save_dir,
                doctor_num=config["doctor_num"],
                max_round=config["max_round"]
            )
            current_report, _, pt, ct = collaboration.collaborate()
            json.dump({"final_report": current_report}, open(f"{sub_save_dir}/meta_final_summary.json", "w"))
            prompt_tokens += pt
            completion_tokens += ct

            # 修正 logits
            logits, pt, ct = leader_agent.revise_logits(analysis)
            if config["ehr_task"] == "outcome":
                logits['ground_truth'] = test_y[patient_index][-1][0]
            else:
                logits['ground_truth'] = test_y[patient_index][-1][2]
            json.dump(logits, open(f"{sub_save_dir}/leader_logits_json.json", "w"))
            prompt_tokens += pt
            completion_tokens += ct

            end = time.time()
            result_json[patient_id] = {
                "report": current_report,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "time": f"{end - start:.2f} s"
            }

            all_prompt_tokens += prompt_tokens
            all_completion_tokens += completion_tokens

        except Exception as e:
            print(f"Error in patient {patient_id}")
            tb.print_exc()
            result_json[patient_id] = "Error"
            continue

    total_end = time.time()
    result_json["total_prompt_tokens"] = all_prompt_tokens
    result_json["total_completion_tokens"] = all_completion_tokens
    result_json["total_time"] = f"{total_end - total_start:.2f} s"

    # 保存全局结果
    with open(f"{save_dir}/result.json", "w") as f:
        json.dump(result_json, f)

    print(f"{len(result_json) - 3} patients have been processed.")
    return result_json