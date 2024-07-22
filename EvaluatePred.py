from difflib import SequenceMatcher
import re


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



def get_score(pred,groundtruth,full_groundtruth):
    exact_match,edit_similarity,identifier_match=eval_pred(pred,groundtruth,full_groundtruth)
    scores = {
        'exact_match': exact_match,
        'edit_similarity': edit_similarity,
        'identifier_match': identifier_match,
        'groundtruth_valid': extract_first_two_code_lines(full_groundtruth)
    }
    return scores