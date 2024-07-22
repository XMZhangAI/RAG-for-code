'''从 cceval 的 JSON 文件中读取测试数据，利用之前的工具检索增强、生成，利用 VLLM 获取生成结果，计算 exact_match、 edit_similarity 等指标。'''
from transformers import AutoTokenizer, AutoModelForCausalLM
import BlocksCutting,FunctionsRetrieval
import json
import re
import base64
import requests
import argparse
import torch
from EvaluatePred import get_score

def load_test_data(jsonl_file):
    '''line_completion.jsonl 将其中的 从begin到end 的数据转化为 python 数据。'''
    data = []
    with open(jsonl_file, 'r') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data

def process_prompt_to_query(prompt: str) -> str:
    last_def_index = prompt.rfind("def")
    if last_def_index != -1:
        query = prompt[last_def_index:]
    else:
        tokens = prompt.split()  # 分词
        if len(tokens) > 30:
            last_tokens = tokens[-30:]  # 获取最后30个 token
        else:
            last_tokens = tokens  # 如果总 token 数小于30，则返回所有 token
        query = ' '.join(last_tokens)  # 将 token 拼接成字符串
    return query

def find_repository_url(repository:str):
    '''在 github 中找到库，返回库的地址'''
    if not repository:
        raise ValueError("Repository not found in metadata")
    
    # 分割 repository 字段以获取 owner 和 repo 名称
    owner_repo1,owner_repo2, version_hash = repository.split("-", 2)
    
    # 构建完整的 GitHub URL
    base_url = "https://github.com/"
    repo_url = f"{base_url}{owner_repo1}/{owner_repo2}"
    return repo_url

GITHUB_TOKEN = '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'

def get_repo_files(repo_url, branch='master'):
    owner_repo = repo_url.replace("https://github.com/", "")
    api_url = f"https://api.github.com/repos/{owner_repo}/git/trees/{branch}?recursive=1"
    headers = {'Authorization': f'token {GITHUB_TOKEN}'}
    response = requests.get(api_url, headers=headers)
    if response.status_code == 404:
        print(f"Branch '{branch}' not found. Trying 'main' branch.")
        api_url = f"https://api.github.com/repos/{owner_repo}/git/trees/master?recursive=1"
        response = requests.get(api_url, headers=headers)
    response.raise_for_status()
    return response.json()

def get_file_content(repo_url, file_path):
    owner_repo = repo_url.replace("https://github.com/", "")
    api_url = f"https://api.github.com/repos/{owner_repo}/contents/{file_path}"
    headers = {'Authorization': f'token {GITHUB_TOKEN}'}
    response = requests.get(api_url, headers=headers)
    response.raise_for_status()
    file_content = response.json()
    return base64.b64decode(file_content['content']).decode('utf-8')

def Cut_and_Retrieve(query,metadata):
    repo_url=find_repository_url(metadata["repository"])
    print(repo_url)
    files_data = get_repo_files(repo_url)
    python_files = [file['path'] for file in files_data['tree'] if file['path'].endswith('.py')]
    file_contents = dict()
    for python_file in python_files:
        content = get_file_content(repo_url, python_file)
        file_contents[python_file]=content
    # 调用 BlocksCutting 和 FunctionsRetrieval
    BlocksCutting.BC_main(file_contents)
    information = FunctionsRetrieval.run_FR("json_temp.json",query,file_contents)

    def get_groundtruth_rightcontext(line1,line2):
            this_content=file_contents[metadata["file"]]
            try:
                lines = this_content.split('\n')
                return lines[line1],'\n'.join(lines[line2:])
            except Exception as e:
                print(f"Error processing file content. Error: {e}")
                return "",""

    ground_truth,right_context=get_groundtruth_rightcontext(metadata['groundtruth_start_lineno'],metadata['right_context_start_lineno'])
    return information,ground_truth,right_context




def join_groundtruth_and_context(groundtruth, right_context):
    """确保groundtruth和right context被正确连接为字符串。"""
    # 如果groundtruth或right context不是字符串，则尝试将其内容转换为字符串
    if not isinstance(groundtruth, str):
        groundtruth = ' '.join(groundtruth) if isinstance(groundtruth, list) else str(groundtruth)
    if not isinstance(right_context, str):
        right_context = ' '.join(right_context) if isinstance(right_context, list) else str(right_context)
    return groundtruth + right_context


def run_model(data,tokenizer,model):
    experiment_result = []
    test_id=1
    for item in data:
        query=process_prompt_to_query(item["prompt"])
        related_information,groundtruth,right_context=Cut_and_Retrieve(query,item["metadata"])

        def format_completion_task(left_context, query,related_code):
            """格式化代码补全任务的输入"""
            formatted_input = (
                f"请根据\n{related_code}\n\n 补全这个函数\n{query}\n\n 要求: 打印输出残缺的那一行的代码和代码的下文.\n"
            )
            return formatted_input

        '''调用大模型'''
        #print(item["metadata"])
        print("ground_truth=",groundtruth)
        
        input_text = format_completion_task(item["prompt"], query, related_information)
        #print("input_text",input_text)
        #input_text=query
        inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
        outputs = model.generate(**inputs, max_new_tokens=140)
        pred = tokenizer.decode(outputs[0],skip_special_tokens=True)
        #print("query=",query)
        # 移除输入文本部分，保留补全内容
        if pred.startswith(input_text):
            pred = pred[len(input_text):]
        print("output begin","#"*80)
        #print("output = ",pred)
        print("output end","#"*80)
        full_groundtruth = join_groundtruth_and_context(groundtruth, right_context)
        scores=get_score(pred,groundtruth,full_groundtruth)
        experiment_result.append(scores)
        print("test",test_id,"scores:",scores)
        test_id+=1
        break
    return experiment_result

def save_to_json(data, output_file):
    """将数据保存为 JSON 文件"""
    with open(output_file, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4)

def main():
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        '--data_path', type=str, default="F:/line_completion.jsonl",
        help='path to directory where data is organized in lang/task.jsonl format'
    )
    parser.add_argument('--temperature', type=float, default=0.2)

    parser.add_argument('--top_p', type=float, default=0.95)
    '''
    parser.add_argument(
        '--task', type=str, required=True,
    )
    '''
    parser.add_argument(
        '--output_file', type=str, default="json_scores.json",
        help='path to directory where to store outputs'
    )
    parser.add_argument(
        '--model', type=str, default="deepseek-ai/deepseek-coder-1.3b-base",
        help='vLLM-supported model'
    )
    parser.add_argument(
        '--tp_size', type=int, default=1,
        help='tensor parallel size'
    )
    parser.add_argument(
        '--model_max_tokens', type=int, default=16384,
        help='maximum number of tokens of the model'
    )
    parser.add_argument(
        '--crossfile_max_tokens', type=int, default=12800,
        help='maximum number of tokens for cross file context'
    )
    parser.add_argument(
        '--generation_max_tokens', type=int, default=50,
        help='maximum number of tokens to generate'
    )
    
    args = parser.parse_args()
    print(json.dumps(vars(args), indent=4))

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, trust_remote_code=True, torch_dtype=torch.bfloat16).cuda()
    data=load_test_data(args.data_path)
    experiment_result=run_model(data,tokenizer,model)
    save_to_json(experiment_result,args.output_file)

main()