'''从 cceval 的 JSON 文件中读取测试数据，利用之前的工具检索增强、生成，利用 VLLM 获取生成结果，计算 exact_match、 edit_similarity 等指标。'''
from transformers import AutoTokenizer, AutoModelForCausalLM
import BlocksCutting,FunctionsRetrieval
import json
import base64
import requests
import argparse
import torch
from get_model_output import Run
from GetInput import get_input

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


GITHUB_TOKEN = 'Your API'

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
    information = FunctionsRetrieval.run_FR("json_temp.json",query,file_contents,"bm25",3)

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

def get_Model_input(data):
    test_id=1
    all_input=[]
    for item in data:
        query=process_prompt_to_query(item["prompt"])
        related_information,groundtruth,right_context=Cut_and_Retrieve(query,item["metadata"])
        '''调用大模型'''
        input_text = get_input(related_information,item["prompt"])
        one_line={
            'input':input_text,
            'groundtruth':groundtruth,
            'right_context':right_context
        }
        all_input.append(one_line)
        if related_information:
            print("test",test_id)
            test_id+=1
        if test_id>2:
            break
        continue
        
        
    with open('output.jsonl', 'w', encoding='utf-8') as f:
        for item in all_input:
            json_string = json.dumps(item, ensure_ascii=False)  # 将字典转换为 JSON 字符串
            f.write(json_string + '\n')
    return 'output.jsonl'



def main():
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        '--data_path', type=str, default="F:/line_completion.jsonl",
        help='path to directory where data is organized in lang/task.jsonl format'
    )
    parser.add_argument('--temperature', type=float, default=0.6)

    parser.add_argument('--top_p', type=float, default=0.95)
    parser.add_argument('--top_k', type=float, default=50)
    parser.add_argument('--do_sample', type=bool, default=True)
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
        '--generation_max_tokens', type=int, default=5000,
        help='maximum number of tokens to generate'
    )
    
    args = parser.parse_args()
    #print(json.dumps(vars(args), indent=4))

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, trust_remote_code=True, torch_dtype=torch.bfloat16).cuda()
    data=load_test_data(args.data_path)
    jsonl_file=get_Model_input(data)
    score_file=Run(tokenizer,model,args,jsonl_file)

main()