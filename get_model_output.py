import json
from EvaluatePred import get_score
from transformers import AutoTokenizer, AutoModelForCausalLM,AutoModel

def save_to_json(data, score_file):
    """将数据保存为 JSON 文件"""
    with open(score_file, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4)


def join_groundtruth_and_context(groundtruth, right_context):
    """确保groundtruth和right context被正确连接为字符串。"""
    # 如果groundtruth或right context不是字符串，则尝试将其内容转换为字符串
    if not isinstance(groundtruth, str):
        groundtruth = ' '.join(groundtruth) if isinstance(groundtruth, list) else str(groundtruth)
    if not isinstance(right_context, str):
        right_context = ' '.join(right_context) if isinstance(right_context, list) else str(right_context)
    return groundtruth + right_context

def Run(args,jsonl_file):
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(args.model_path, local_files_only=True)    #tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    #model = AutoModelForCausalLM.from_pretrained(args.model, trust_remote_code=True, torch_dtype=torch.bfloat16).cuda()
    experiment_result=[]
    data = []
    with open(jsonl_file, 'r') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    for item in data:
        input_text,groundtruth,right_context=item['input'],item['groundtruth'],item['right_context']
        print("ok"*100)
        input_text="你是一个法语老师，请将“我爱你”翻译成法语！"
        inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
        
        outputs = model.generate(**inputs, do_sample=args.do_sample, temperature=args.temperature, top_p=args.top_p,max_new_tokens=args.generation_max_tokens)
        
        pred = tokenizer.decode(outputs[0], skip_special_tokens=True)
        if pred.startswith(input_text):
            pred = pred[len(input_text):]
        print("output begin","#"*80)
        print(pred)
        print("output end","#"*80)
        full_groundtruth = join_groundtruth_and_context(groundtruth, right_context)
        scores=get_score(pred,groundtruth,full_groundtruth)
        experiment_result.append(scores)

    save_to_json(experiment_result,'score_file.json')

