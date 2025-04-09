import asyncio
import re

import numpy as np
from typing import List, Dict, Any
import json

from swift.plugin import ORM, orms
from swift.utils import get_logger

logger = get_logger()


# Code borrowed from plugin/orm.py
class MathAccuracy(ORM):

    def __init__(self):
        import importlib.util
        assert importlib.util.find_spec('math_verify') is not None, (
            "The math_verify package is required but not installed. Please install it using 'pip install math_verify'.")

    def __call__(self, completions, solution, **kwargs) -> List[float]:
        from latex2sympy2_extended import NormalizationConfig
        from math_verify import LatexExtractionConfig, parse, verify
        rewards = []
        for content, sol in zip(completions, solution):
            gold_parsed = parse(sol, extraction_mode='first_match', extraction_config=[LatexExtractionConfig()])
            if len(gold_parsed) != 0:
                # We require the answer to be provided in correct latex (no malformed operators)
                answer_parsed = parse(
                    content,
                    extraction_config=[
                        LatexExtractionConfig(
                            normalization_config=NormalizationConfig(
                                nits=False,
                                malformed_operators=False,
                                basic_latex=True,
                                equations=True,
                                boxed=True,
                                units=True,
                            ),
                            # Ensures that boxed is tried first
                            boxed_match_priority=0,
                            try_extract_without_anchor=False,
                        )
                    ],
                    extraction_mode='first_match',
                )
                # Reward 1 if the content is the same as the ground truth, 0 otherwise
                reward = float(verify(answer_parsed, gold_parsed))
            else:
                # If the gold solution is not parseable, we reward 1 to skip this example
                reward = 1.0
            rewards.append(reward)
        return rewards


class MathFormat(ORM):

    def __call__(self, completions, **kwargs) -> List[float]:
        """Reward function that checks if the completion has a specific format."""
        pattern = r'^<think>.*?</think>\s*<answer>.*?</answer>(?![\s\S])'
        matches = [re.match(pattern, content, re.DOTALL | re.MULTILINE) for content in completions]
        return [1.0 if match else 0.0 for match in matches]


class CountdownORM(ORM):

    def __call__(self, completions, target, nums, **kwargs) -> List[float]:
        """
        Evaluates completions based on Mathematical correctness of the answer

        Args:
            completions (list[str]): Generated outputs
            target (list[str]): Expected answers
            nums (list[str]): Available numbers

        Returns:
            list[float]: Reward scores
        """
        rewards = []
        for completion, gt, numbers in zip(completions, target, nums):
            try:
                # Check if the format is correct
                match = re.search(r'<answer>(.*?)<\/answer>', completion)
                if match is None:
                    rewards.append(0.0)
                    continue
                # Extract the "answer" part from the completion
                equation = match.group(1).strip()
                if '=' in equation:
                    equation = equation.split('=')[0]
                # Extract all numbers from the equation
                used_numbers = [int(n) for n in re.findall(r'\d+', equation)]

                # Check if all numbers are used exactly once
                if sorted(used_numbers) != sorted(numbers):
                    rewards.append(0.0)
                    continue
                # Define a regex pattern that only allows numbers, operators, parentheses, and whitespace
                allowed_pattern = r'^[\d+\-*/().\s]+$'
                if not re.match(allowed_pattern, equation):
                    rewards.append(0.0)
                    continue

                # Evaluate the equation with restricted globals and locals
                result = eval(equation, {"__builti'ns__": None}, {})
                # Check if the equation is correct and matches the ground truth
                if abs(float(result) - float(gt)) < 1e-5:
                    rewards.append(1.0)
                else:
                    rewards.append(0.0)
            except Exception:
                # If evaluation fails, reward is 0
                rewards.append(0.0)
        return rewards


class MultiModalAccuracyORM(ORM):

    def __call__(self, completions, solution, **kwargs) -> List[float]:
        """
        Reward function that checks if the completion is correct.
        Args:
            completions (list[str]): Generated outputs
            solution (list[str]): Ground Truths.

        Returns:
            list[float]: Reward scores
        """
        rewards = []
        from math_verify import parse, verify
        for content, sol in zip(completions, solution):
            reward = 0.0
            # Try symbolic verification first
            try:
                answer = parse(content)
                if float(verify(answer, parse(sol))) > 0:
                    reward = 1.0
            except Exception:
                pass  # Continue to next verification method if this fails

            # If symbolic verification failed, try string matching
            if reward == 0.0:
                try:
                    # Extract answer from solution if it has think/answer tags
                    sol_match = re.search(r'<answer>(.*?)</answer>', sol)
                    ground_truth = sol_match.group(1).strip() if sol_match else sol.strip()

                    # Extract answer from content if it has think/answer tags
                    content_match = re.search(r'<answer>(.*?)</answer>', content)
                    student_answer = content_match.group(1).strip() if content_match else content.strip()

                    # Compare the extracted answers
                    if student_answer == ground_truth:
                        reward = 1.0
                except Exception:
                    pass  # Keep reward as 0.0 if both methods fail
            rewards.append(reward)
        return rewards


# ref implementation: https://github.com/huggingface/open-r1/blob/main/src/open_r1/rewards.py
class CodeReward(ORM):

    def __init__(self):
        import importlib.util
        assert importlib.util.find_spec('e2b') is not None, (
            "The e2b package is required but not installed. Please install it using 'pip install e2b-code-interpreter'."
        )
        from dotenv import load_dotenv
        load_dotenv()

    @staticmethod
    def extract_code(completion: str, language: str) -> str:
        pattern = re.compile(rf'```{language}\n(.*?)```', re.DOTALL)
        matches = pattern.findall(completion)
        extracted_answer = matches[-1] if len(matches) >= 1 else ''
        return extracted_answer

    def run_async_from_sync(self, scripts: List[str], languages: List[str]) -> List[float]:
        """Function wrapping the `run_async` function."""
        # Create a new event loop and set it
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            # Run the async function and get the result
            rewards = loop.run_until_complete(self.run_async(scripts, languages))
        finally:
            loop.close()

        return rewards

    async def run_async(self, scripts: List[str], languages: List[str]) -> List[float]:
        from e2b_code_interpreter import AsyncSandbox

        # Create the sandbox by hand, currently there's no context manager for this version
        try:
            sbx = await AsyncSandbox.create(timeout=30, request_timeout=3)
        except Exception as e:
            logger.warning(f'Error from E2B executor: {e}')
            return [0.0] * len(scripts)
        # Create a list of tasks for running scripts concurrently
        tasks = [self.run_script(sbx, script, language) for script, language in zip(scripts, languages)]

        # Wait for all tasks to complete and gather their results as they finish
        results = await asyncio.gather(*tasks)
        rewards = list(results)  # collect results

        # Kill the sandbox after all the tasks are complete
        await sbx.kill()

        return rewards

    async def run_script(self, sbx, script: str, language: str) -> float:
        try:
            execution = await sbx.run_code(script, language=language, timeout=30)
        except Exception as e:
            logger.warning(f'Error from E2B executor: {e}')
            return 0.0
        try:
            return float(execution.text)
        except (TypeError, ValueError):
            return 0.0

    def __call__(self, completions, **kwargs) -> List[float]:
        """Reward function that evaluates code snippets using the E2B code interpreter.

        Assumes the dataset contains a `verification_info` column with test cases.
        """
        evaluation_script_template = """
        import subprocess
        import json

        def evaluate_code(code, test_cases):
            passed = 0
            total = len(test_cases)
            exec_timeout = 5

            for case in test_cases:
                process = subprocess.run(
                    ["python3", "-c", code],
                    input=case["input"],
                    text=True,
                    capture_output=True,
                    timeout=exec_timeout
                )

                if process.returncode != 0:  # Error in execution
                    continue

                output = process.stdout.strip()
                if output.strip() == case["output"].strip():
                    passed += 1

            success_rate = (passed / total)
            return success_rate

        code_snippet = {code}
        test_cases = json.loads({test_cases})

        evaluate_code(code_snippet, test_cases)
        """
        verification_info = kwargs['verification_info']
        languages = [info['language'] for info in verification_info]
        code_snippets = [
            self.extract_code(completion, language) for completion, language in zip(completions, languages)
        ]
        scripts = [
            evaluation_script_template.format(
                code=json.dumps(code), test_cases=json.dumps(json.dumps(info['test_cases'])))
            for code, info in zip(code_snippets, verification_info)
        ]
        try:
            rewards = self.run_async_from_sync(scripts, languages)

        except Exception as e:
            logger.warning(f'Error from E2B executor: {e}')
            rewards = [0.0] * len(completions)

        return rewards


class CodeFormat(ORM):

    def __call__(self, completions, **kwargs) -> List[float]:
        verification_info = kwargs['verification_info']
        rewards = []
        for content, info in zip(completions, verification_info):
            pattern = r'^<think>.*?</think>\s*<answer>.*?```{}.*?```.*?</answer>(?![\s\S])'.format(info['language'])
            match = re.match(pattern, content, re.DOTALL | re.MULTILINE)
            reward = 1.0 if match else 0.0
            rewards.append(reward)
        return rewards


class EbitdaPredictionORM(ORM):
    """
    用于评估EBITDA预测结果的Outcome Reward Model (ORM)。
    奖励基于预测值与真实值的接近程度（例如，MAE的倒数）。
    """

    def parse_answer(self, text: str) -> List[float] | None:
        """从模型的输出中解析<answer>标签内的预测值"""
        # 使用re.DOTALL允许多行匹配，以防<answer>标签跨行
        match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
        if not match:
            print("ORM_DEBUG: No <answer> tag found.")
            return None
        try:
            content = match.group(1).strip()
            # 移除可能的方括号
            if content.startswith('[') and content.endswith(']'):
                content = content[1:-1]
            # 按逗号分割并转换为浮点数
            predictions = [float(x.strip()) for x in content.split(',')]
            # 期望正好有4个预测值
            if len(predictions) == 4:
                return predictions
            else:
                print(f"ORM_DEBUG: Incorrect number of predictions found: {len(predictions)}. Expected 4.")
                return None # 预测数量不符
        except ValueError as e:
            print(f"ORM_DEBUG: Error parsing prediction values: {e}. Content: '{match.group(1)}'")
            return None # 无法解析为浮点数
        except Exception as e:
            print(f"ORM_DEBUG: Unexpected error during parsing: {e}")
            return None

    def calculate_reward(self, predictions: List[float] | None, ground_truths: List[float]) -> float:
        """根据预测值和真实值计算奖励分数"""
        if predictions is None:
            # 如果格式错误或无法解析，给予最低奖励
            return 0.0

        if len(predictions) != len(ground_truths):
             print(f"ORM_DEBUG: Mismatch between prediction ({len(predictions)}) and ground truth ({len(ground_truths)}) lengths.")
             return 0.0 # 理论上不应发生，因为前面检查了长度，但作为保险

        try:
            # 计算 Mean Absolute Error (MAE)
            # mae = np.mean(np.abs(np.array(predictions) - np.array(ground_truths)))

            # 将MAE转换为奖励分数 (0到1之间)，MAE越小，奖励越高
            # 使用 1 / (1 + MAE) 的形式，确保MAE=0时奖励为1，MAE越大奖励越趋近于0
            # 添加一个小的epsilon防止MAE非常接近0时分母过小或为0 (虽然MAE非负)
            # epsilon = 1e-9
            # reward = 1.0 / (1.0 + mae)

            # --- 可选：考虑其他奖励机制 ---
            # 1. 基于MAPE (Mean Absolute Percentage Error)，对规模不敏感，但需处理真实值为0的情况
            truths_array = np.array(ground_truths)
            if np.any(truths_array == 0):
                # 处理分母为0的情况，使用 MAE 或给一个固定惩罚（概率很小）
                mae = np.mean(np.abs(np.array(predictions) - np.array(ground_truths)))
                reward = 1.0 / (1.0 + mae)
            else:
                mape = np.mean(np.abs((np.array(predictions) - truths_array) / truths_array)) * 100
                reward = max(0.0, 1.0 - mape / 100.0) # 简单线性转换，MAPE=100%时奖励为0

            # 2. 考虑加入对<think>标签存在的奖励 (简单存在性检查)
            # if '<think>' in completion_text and '</think>' in completion_text:
            #    reward = min(1.0, reward + 0.05) # 给少量奖励加成

            return reward

        except Exception as e:
            print(f"ORM_DEBUG: 计算奖励出错: {e}")
            return 0.0 # 计算出错则返回0奖励

    def __call__(self, completions: List[str], ground_truth_ebitda: List[List[float]], **kwargs) -> List[float]:
        """
        主调用函数，处理一批次的生成结果和真实标签。

        Args:
            completions (List[str]): 模型生成的完整文本列表。
            ground_truth_ebitda (List[List[float]]): 对应的真实未来4季度EBITDA列表 (从数据集中透传)。
            **kwargs: 其他从数据集中透传的字段 (例如 company_id, last_quarter)。

        Returns:
            List[float]: 每个生成结果对应的奖励分数列表。
        """
        rewards = []
        if len(completions) != len(ground_truth_ebitda):
             print(f"ORM_ERROR: Mismatch in length between completions ({len(completions)}) and ground_truth_ebitda ({len(ground_truth_ebitda)})!")
             # 返回与completions等长的0奖励列表，或抛出错误
             return [0.0] * len(completions)

        print(f"ORM_INFO: 处理 {len(completions)} 个completions.") # 添加日志方便调试

        for i in range(len(completions)):
            completion_text = completions[i]
            truth = ground_truth_ebitda[i]

            # 1. 解析预测值
            predicted_values = self.parse_answer(completion_text)
            if predicted_values:
                 print(f"ORM_DEBUG: 解析成功 {i}: {predicted_values}")
            else:
                 print(f"ORM_DEBUG: 解析失败 {i} 模型输出: ...'{completion_text[-200:]}'") # 打印部分文本以供调试

            # 2. 计算奖励
            reward = self.calculate_reward(predicted_values, truth)
            print(f"ORM_DEBUG: 奖励函数输出： {i}: {reward}")

            rewards.append(reward)

        return rewards



class ProgressiveFormatORM(ORM):
    """
    用于评估模型输出格式的渐进式奖励函数。
    通过分阶段奖励不同的格式完成度，引导模型逐步掌握 <think></think><answer></answer> 的回答模式。
    """

    def evaluate_format(self, text: str) -> float:
        """
        评估文本格式的完成度，返回0.0到1.0之间的奖励分数。
        
        奖励等级:
        1. 0.0: 没有任何格式元素
        2. 0.2: 包含 <think> 标签
        3. 0.4: 包含 <think> 和 </think> 标签
        4. 0.6: 包含 <think></think> 且有内容
        5. 0.8: 包含 <think></think> 和 <answer> 标签
        6. 1.0: 包含完整的 <think></think><answer></answer> 格式
        """
        # 检查是否包含完整的格式
        full_pattern = r'^<think>.*?</think>\s*<answer>.*?</answer>(?![\s\S])'
        if re.match(full_pattern, text, re.DOTALL | re.MULTILINE):
            return 1.0
            
        # 检查是否包含 <think></think> 和 <answer> 标签
        partial_pattern = r'<think>.*?</think>\s*<answer>'
        if re.search(partial_pattern, text, re.DOTALL | re.MULTILINE):
            return 0.8
            
        # 检查是否包含 <think></think> 且有内容
        think_content_pattern = r'<think>.*?</think>'
        think_match = re.search(think_content_pattern, text, re.DOTALL | re.MULTILINE)
        if think_match and len(think_match.group(0)) > len('<think></think>'):
            return 0.6
            
        # 检查是否包含 <think> 和 </think> 标签
        if '<think>' in text and '</think>' in text:
            return 0.4
            
        # 检查是否只包含 <think> 标签
        if '<think>' in text:
            return 0.2
            
        # 没有任何格式元素
        return 0.0

    def __call__(self, completions: List[str], **kwargs) -> List[float]:
        """
        主调用函数，处理一批次的生成结果。

        Args:
            completions (List[str]): 模型生成的完整文本列表。
            **kwargs: 其他参数。

        Returns:
            List[float]: 每个生成结果对应的格式奖励分数列表。
        """
        rewards = []
        print(f"ORM_INFO: 处理 {len(completions)} 个completions的格式评估.")
        
        for i, completion_text in enumerate(completions):
            # 评估格式
            reward = self.evaluate_format(completion_text)
            print(f"ORM_DEBUG: 格式评估 {i}: {reward}")
            rewards.append(reward)
            
        return rewards



orms['external_math_acc'] = MathAccuracy
orms['external_math_format'] = MathFormat
orms['external_countdown'] = CountdownORM
orms['external_r1v_acc'] = MultiModalAccuracyORM
orms['external_code_reward'] = CodeReward
orms['external_code_format'] = CodeFormat

orms['external_ebitda_predictor'] = EbitdaPredictionORM 
orms['external_progressive_format'] = ProgressiveFormatORM 