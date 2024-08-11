from difflib import SequenceMatcher
import re
import json

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

def join_groundtruth_and_context(groundtruth, right_context):
    """确保groundtruth和right context被正确连接为字符串。"""
    # 如果groundtruth或right context不是字符串，则尝试将其内容转换为字符串
    if not isinstance(groundtruth, str):
        groundtruth = ' '.join(groundtruth) if isinstance(groundtruth, list) else str(groundtruth)
    if not isinstance(right_context, str):
        right_context = ' '.join(right_context) if isinstance(right_context, list) else str(right_context)
    return groundtruth + right_context

def eval_pred(pred,groundT,full_groundtruth):
    #print(groundT)
    exact_match=calculate_exact_match(pred,groundT)
    edit_similarity=calculate_edit_similarity(pred,full_groundtruth)
    identifier_similarity=calculate_identifier_match(pred,full_groundtruth)

    return exact_match,edit_similarity,identifier_similarity



def get_score(pred,groundtruth,full_groundtruth):
    exact_match,edit_similarity,identifier_match=eval_pred(pred,groundtruth,full_groundtruth)
    scores = {
        'exact_match': exact_match,
        'edit_similarity': edit_similarity,
        'identifier_match': identifier_match,
        'groundtruth_valid': extract_first_two_code_lines(full_groundtruth)
    }
    return scores


def load_score(json_file,scores_result):
    data=[]
    results=[]
    
    with open(json_file, 'r') as f1:
        for line in f1:
            data.append(json.loads(line.strip()))
    
    for d in data:
        #print(d['task_id'][0])
        ground_truth=d['groundtruth'][0]
        full_groundtruth=join_groundtruth_and_context(ground_truth,d['right_context'])
        score=get_score(d['pred'],ground_truth,full_groundtruth)
        result={
            'score':score,
            'idx':d['task_id'][0]
        }
        results.append(result)

    with open(scores_result, 'w') as f2:
        for ret in results:
            json_string = json.dumps(ret, ensure_ascii=False)  # 将字典转换为 JSON 字符串
            f2.write(json_string + '\n')