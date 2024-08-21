"""
这里并不套用模版！！！
"""
import os
import torch
import warnings
warnings.filterwarnings("ignore", message="`num_beams` is set to 1. However, `length_penalty` is set to `1.6`")

# 设置 Hugging Face 的离线模式
os.environ['HF_DATASETS_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

import argparse
import json
from typing import List

from tqdm import tqdm
from transformers import AutoTokenizer
from transformers.utils import logging
from vllm import LLM, SamplingParams # type: ignore

logging.set_verbosity_info()
logger = logging.get_logger(__name__)
# add a small buffer to take care of non-lossless tokenizers
BUFFER = 100


def truncate(prompt: str, max_num_tokens: int, side: str, tokenizer) -> str:
    """Truncate prompt from side given the token budget"""

    tokens = tokenizer.tokenize(prompt)
    num_tokens = len(tokens)

    if num_tokens > max_num_tokens:
        if side == 'left':
            prompt_tokens = tokens[num_tokens - max_num_tokens:]
        elif side == 'right':
            prompt_tokens = tokens[:max_num_tokens]
        prompt = tokenizer.convert_tokens_to_string(prompt_tokens)
        new_len = len(tokenizer.tokenize(prompt))
        if new_len > max_num_tokens:
            logger.warning(
                f'Number of tokens after truncation is greater than max tokens allowed: {new_len=} {num_tokens=}')
    return prompt


def prepare_prompt(
        prompt: str,
        additional_context: str,
        cross_file_budget: int,
        prompt_budget: int,
        tokenizer
) -> str:
    """Create an augmented prompt according to budget specs"""

    # print(f'{cross_file_budget=} {prompt_budget=}')
    # left truncate original prompt
    prompt = truncate(prompt, prompt_budget, 'left', tokenizer)

    if additional_context is not None:
        # right truncate cross file context string
        additional_context = truncate(additional_context, cross_file_budget, 'right', tokenizer)
    else:
        additional_context = ''

    return additional_context + '\n' + prompt

def cceval_generate(
        args,
        data,
        tokenizer,
        sampling_params,
        llm,
        out_path
) -> List[str]:
    all_prompts = []
    for d in data:
        
        base_prompt = d['base_prompt']
        contexts = d['similar_function']
        retrieve_functions=""
        for context in contexts:
            retrieve_functions+=context
        prompt = prepare_prompt(
                base_prompt, retrieve_functions,
                args.crossfile_max_tokens,
                args.model_max_tokens - args.generation_max_tokens - args.crossfile_max_tokens - BUFFER,
                tokenizer
            )
        all_prompts.append(prompt)
    outputs= llm.generate(all_prompts, sampling_params)

    with open(out_path, 'w') as f:
        idx = 0
        for d in data:
            d_base = d.copy()
            d_base['pred'] = outputs[idx].outputs[0].text
            d_base['task_id'] = d['task_id'],
            d_base['groundtruth']=d['groundtruth'],
            #print(d_base['groundtruth'])
            d_base['right_context']=d['right_context']
            json_string = json.dumps(d_base, ensure_ascii=False)  # 将字典转换为 JSON 字符串
            f.write(json_string + '\n')
            idx += 1
    print(idx,"length=",len(all_prompts))
    return



def vllm_infer_run(model_path, args,input_jsonl_file):
    # 确保模型和分词器从本地加载
    llm = LLM(model=model_path, tensor_parallel_size=args.tp_size,trust_remote_code=True, max_model_len=args.model_max_tokens)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True,trust_remote_code=True)
    sampling_params = SamplingParams(temperature=args.temperature, top_p=args.top_p, max_tokens=args.generation_max_tokens)

    # setup path
    out_path = 'reply_prediction(try).jsonl'
    data = []
    with open(input_jsonl_file, 'r') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    # generation
    cceval_generate(args, data, tokenizer, sampling_params, llm,out_path)
    return out_path


'''处理输入的时候需要注意：模型的输出结果应该是包括 groundtruth 的完整的一行。'''