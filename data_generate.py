import re
from datasets import load_dataset, Dataset
from typing import Dict, Any

# --- 复用 Unsloth 中的辅助函数 ---
def extract_hash_answer(text: str) -> str | None:
    """从 GSM8K 原始答案中提取 #### 后面的数字答案"""
    if "####" not in text:
        return None
    return text.split("####")[1].strip()

def extract_xml_answer(text: str) -> str:
    """从模型的 XML 格式输出中提取 <answer> 标签内的内容"""
    # 增加健壮性，处理找不到标签或格式错误的情况
    if "<answer>" not in text or "</answer>" not in text:
        return "" # 或者返回 None，根据下游处理逻辑决定
    try:
        answer = text.split("<answer>")[-1]
        answer = answer.split("</answer>")[0]
        return answer.strip()
    except Exception:
        return "" # 解析出错也返回空字符串或 None

# --- MS-Swift GRPO 数据集准备 ---

# 定义系统提示 (与 Unsloth 保持一致)
SYSTEM_PROMPT = """
Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""

def format_gsm8k_for_grpo(example: Dict[str, Any]) -> Dict[str, Any]:
    """将单个 GSM8K 样本转换为 MS-Swift GRPO 格式"""
    question = example['question']
    raw_answer_text = example['answer']
    ground_truth_answer = extract_hash_answer(raw_answer_text)

    # GRPO 的 messages 只包含 prompt 部分
    messages = [
        {'role': 'system', 'content': SYSTEM_PROMPT},
        {'role': 'user', 'content': question}
    ]

    # 过滤掉无法提取答案的样本 (可选，但推荐)
    if ground_truth_answer is None:
        return None # 或者返回一个特殊标记以便后续过滤

    return {
        'messages': messages,
        'ground_truth_answer': ground_truth_answer # 将真实答案放在这里，以便透传给 ORM
        # 可以添加其他需要透传的字段，例如 'original_question': question
    }

def get_gsm8k_swift_grpo_dataset(split: str = "train") -> Dataset:
    """加载并准备 GSM8K 数据集以用于 MS-Swift GRPO"""
    print(f"Loading GSM8K dataset split: {split}")
    # 注意 'openai/gsm8k' 可能需要特定版本的 datasets 库或身份验证
    try:
        dataset = load_dataset('openai/gsm8k', 'main', split=split)
    except Exception as e:
         print(f"Error loading dataset 'openai/gsm8k': {e}")
         print("Attempting to load 'gsm8k' instead...")
         try:
             # 有些环境可能需要使用 'gsm8k' 这个名字
             dataset = load_dataset('gsm8k', 'main', split=split)
         except Exception as e2:
             print(f"Error loading dataset 'gsm8k': {e2}")
             raise ConnectionError("Failed to load GSM8K dataset from Hugging Face Hub.") from e2


    print("Formatting dataset for MS-Swift GRPO...")
    # 使用 map 进行转换
    formatted_dataset = dataset.map(format_gsm8k_for_grpo, remove_columns=dataset.column_names)

    # 过滤掉转换失败的样本 (如果 format_gsm8k_for_grpo 返回 None)
    # 注意: datasets V2 map() 默认不移除返回 None 的行，需要 filter
    # 如果 map 返回 None，需要 filter 掉
    # formatted_dataset = formatted_dataset.filter(lambda x: x is not None)
    # 或者确保 format_gsm8k_for_grpo 不返回 None，而是引发错误或返回有效字典
    # 如果 format_gsm8k_for_grpo 保证总是返回字典，则无需过滤
    # 为了简单起见，我们假设 extract_hash_answer 总是能找到答案（在gsm8k中通常是这样）
    # 如果要处理 edge cases，需要添加过滤步骤

    print(f"Dataset preparation complete. Number of samples: {len(formatted_dataset)}")
    print("Sample data point:")
    print(formatted_dataset[0])

    return formatted_dataset

# --- 使用示例 ---
if __name__ == "__main__":
    # 准备训练集
    train_dataset = get_gsm8k_swift_grpo_dataset(split="train")
    # 准备测试集 (如果需要)
    test_dataset = get_gsm8k_swift_grpo_dataset(split="test")

    # 可以将处理后的数据集保存到本地，例如 JSON Lines 格式
    output_dir = "gsm8k_swift_grpo"
    import os
    os.makedirs(output_dir, exist_ok=True)

    train_path = os.path.join(output_dir, "train.jsonl")
    test_path = os.path.join(output_dir, "test.jsonl")

    print(f"Saving processed train dataset to {train_path}")
    train_dataset.to_json(train_path, orient="records", lines=True)

    print(f"Saving processed test dataset to {test_path}")
    test_dataset.to_json(test_path, orient="records", lines=True)

    print("Done.")

    # 在 MS-Swift 命令行中，你可以这样使用:
    # --dataset ./gsm8k_swift_grpo/train.jsonl ./gsm8k_swift_grpo/test.jsonl
    # 或者
    # --dataset ./gsm8k_swift_grpo # 如果文件夹下只有这两个文件