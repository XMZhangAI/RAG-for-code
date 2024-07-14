'''从 cceval 的 JSON 文件中读取测试数据，利用之前的工具检索增强、生成，利用 VLLM 获取生成结果，计算 exact_match、 edit_similarity 等指标。
利用 JSON 文件储层结果，简记为 result'''
'''TODO:暂时不知道 right_context 有什么用'''
import json
from difflib import SequenceMatcher
import re

def load_test_data(begin,end,jsonl_file):
    '''line_completion.jsonl 将其中的 从begin到end 的数据转化为 python 数据。'''
    with open(jsonl_file, 'r') as f:
        data = json.load(f)
        return data

def process_prompt_to_query(prompt:str):
    query=None
    return query


def find_repository_path(repository:str):
    '''在 github 中找到库，返回库的地址'''

    path=None
    return path

def Cut_and_Retrieve(query,repo_path):
    information={"left_context":str,"same_class_methods":[],"depend_func":[]}
    '''调用 FunctionRetrieval 和 BlocksCutting'''

    return information

def get_pred(information,model):
    ret=" "
    return ret


def calculate_exact_match(pred, groundtruth):
    """计算精确匹配得分。"""
    return 1 if pred.startswith(groundtruth) else 0

def extract_first_two_code_lines(code):
    """提取前两行有效代码行（排除空行和注释）。"""
    lines = code.split('\n')
    valid_lines = [line for line in lines if line.strip() and not line.strip().startswith('#')]
    return '\n'.join(valid_lines[:2])

def calculate_edit_similarity(pred, groundtruth):
    """计算编辑相似度得分。"""
    pred_code_lines = extract_first_two_code_lines(pred)
    groundtruth_code_lines = extract_first_two_code_lines(groundtruth)
    return SequenceMatcher(None, pred_code_lines, groundtruth_code_lines).ratio()

def extract_identifiers(code):
    """提取代码中的标识符。"""
    return set(re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', code))

def calculate_identifier_match(pred, groundtruth):
    """计算标识符匹配得分，只考虑前两行有效代码。"""
    pred_code_lines = extract_first_two_code_lines(pred)
    groundtruth_code_lines = extract_first_two_code_lines(groundtruth)
    
    pred_identifiers = extract_identifiers(pred_code_lines)
    groundtruth_identifiers = extract_identifiers(groundtruth_code_lines)
    if not pred_identifiers or not groundtruth_identifiers:
        return 0  # 避免除以零
    common_identifiers = pred_identifiers.intersection(groundtruth_identifiers)
    return len(common_identifiers) / len(groundtruth_identifiers)


def eval_pred(pred,groundT,full_groundtruth):

    exact_match=calculate_exact_match(pred,groundT)
    edit_similarity=calculate_edit_similarity(pred,full_groundtruth)
    identifier_similarity=calculate_identifier_match(pred,full_groundtruth)

    return exact_match,edit_similarity,identifier_similarity

def join_groundtruth_and_context(groundtruth, right_context):
    """确保groundtruth和right context被正确连接为字符串。"""
    # 如果groundtruth或right context不是字符串，则尝试将其内容转换为字符串
    if not isinstance(groundtruth, str):
        groundtruth = ' '.join(groundtruth) if isinstance(groundtruth, list) else str(groundtruth)
    if not isinstance(right_context, str):
        right_context = ' '.join(right_context) if isinstance(right_context, list) else str(right_context)
    return groundtruth + right_context


def run_model(model,data):
    experiment_result = []
    for item in data:
        query=process_prompt_to_query(item["prompt"])
        repo=find_repository_path(item["repository"])
        information=Cut_and_Retrieve(query,repo)
        
        '''调用大模型，deepseek-coder 1.5b,'''
        # TODO: finish function

        pred=get_pred(information,model)
        groundtruth=item["groundtruth"]
        right_context=item["right_context"]    
        full_groundtruth = join_groundtruth_and_context(groundtruth, right_context)
        
        exact_match,edit_similarity,identifier_match=eval_pred(pred,groundtruth,full_groundtruth)
        scores = {
            'exact_match': exact_match,
            'edit_similarity': edit_similarity,
            'identifier_match': identifier_match,
            'groundtruth_valid': extract_first_two_code_lines(full_groundtruth)
        }
        
        experiment_result.append(scores)
    return experiment_result

def save_to_json(data, output_file):
    """将数据保存为 JSON 文件"""
    with open(output_file, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4)

def main(start,end,jsonl_file,model,output_file):
    data=load_test_data(start,end,jsonl_file)
    experiment_result=run_model(data,model)

    '''TODO: process experiment_result'''

    save_to_json(experiment_result,output_file)

main(1,100,"cceval_josnl","my_model","output_path")