#TODO: 同时需要注意 给出的 query 文本长度可能比较大，比如由多个函数组成，这样会大大破坏比较的准确性。bm25 考虑到了文本的长度
import json
import os
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer # type: ignore
import numpy as np
from BlocksCutting import function_block
from rank_bm25 import BM25Okapi
import openai
import argparse

class block(function_block):
    def __init__(self, file_path_,name=None, b_class=None, s_line=None, e_line=None, calls=None, import_list=None):
        super().__init__(name, b_class, s_line, e_line, calls, import_list)
        self.file_path=file_path_

    def __repr__(self):
        return (f"FunctionBlock(name={self.name}, class_name={self.belong_class}, "
                f"lineno={self.start_line}, end_lineno={self.end_line}, calls={self.call_func},import={self.import_repo},file_path={self.file_path})")


def load_function_blocks(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)

    function_blocks = []
    class_methods = {}

    for file_path, contents in data.items():
        class_methods[file_path] = {}

        for func in contents["Functions"]:
            function_block = block(
                file_path_=file_path,
                name=func['name'],
                b_class=func['class'],
                s_line=func['lineno'],
                e_line=func['end_lineno'],
                calls=func['calls'],
                import_list=func['import']
            )
            function_blocks.append(function_block)

        for cls in contents["Classes"]:
            class_name = cls['class_name']
            methods = cls['methods']# methods 已经是一个 list
            class_methods[file_path][class_name] = methods

    return function_blocks, class_methods

def load_query(query_file):
    with open(query_file, 'r') as f:
        query = f.read()
    return query


def get_function_text(block):
    file_path = block.file_path
    start_line = block.start_line
    end_line = block.end_line
    
    with open(os.path.join(root_dir, file_path), 'r') as f:
        lines = f.readlines()
    function_text = ''.join(lines[start_line-1:end_line])
    return function_text

'''读取文件，获取文本'''

#def cosine_similarity(vec1, vec2):
#    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


def compute_bm25_similarity(query, function_blocks):
    texts = [get_function_text(block) for block in function_blocks]
    tokenized_texts = [text.split() for text in texts]
    bm25 = BM25Okapi(tokenized_texts)

    query_tokens = query.split()
    scores = bm25.get_scores(query_tokens)
    return scores

def compute_tfidf_similarity(query, function_blocks):
    texts = [get_function_text(block) for block in function_blocks]
    texts.append(query)
    
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(texts)
    
    query_vector = tfidf_matrix[-1]
    function_vectors = tfidf_matrix[:-1]
    #变成向量之后展平
    scores = (function_vectors*query_vector.T).toarray().flatten()
    return scores

def jaccard_similarity(query, document):
    query_set = set(query.split())
    document_set = set(document.split())
    intersection = query_set.intersection(document_set)
    union = query_set.union(document_set)
    return len(intersection) / len(union)

def compute_jaccard_similarity(query, function_blocks):
    similarities = []
    for block in function_blocks:
        text = get_function_text(block)
        similarity = jaccard_similarity(query, text)
        similarities.append(similarity)
    return similarities

#openai.api_key = 'your_openai_api_key'
def get_embedding(text, model="text-embedding-ada-002"):
    response = openai.Embedding.create(
        input=text,
        model=model
    )
    return response['data'][0]['embedding']

def compute_openai_similarity(query, function_blocks):
    query_embedding = get_embedding(query)
    similarities = []
    for block in function_blocks:
        text = get_function_text(block)
        text_embedding = get_embedding(text)
        similarity = cosine_similarity(query_embedding, text_embedding)
        similarities.append(similarity)
    return similarities

def lexical_ranking(
        query,
        docs,
        ranking_fn,#排序方式
        top_n,
        doc_ids=None,#如果为 true 会淘汰非常小的检索结果
        score_threshold=None,
):
    '''这个函数根据不同的排序方法（如 bm25、tfidf、jaccard_sim）对文档进行排序。
    可以选择是否使用评分阈值来过滤文档，并按评分对文档进行排序。'''
    if ranking_fn == "bm25":
        scores = compute_bm25_similarity(query, docs)
    elif ranking_fn == "tfidf":
        scores = compute_tfidf_similarity(query, docs)
    elif ranking_fn == "jaccard_sim":
        scores = compute_jaccard_similarity(query,docs)
    elif ranking_fn == "openai":
        scores=compute_openai_similarity(query,docs)
    else:
        raise NotImplementedError

    if score_threshold is not None:
        #根据阈值筛选
        skip_ids = [idx for idx, s in enumerate(scores) if s < score_threshold]
        scores = [s for idx, s in enumerate(scores) if idx not in skip_ids]
        docs = [d for idx, d in enumerate(docs) if idx not in skip_ids]
        if doc_ids is not None:
            doc_ids = [doc_id for idx, doc_id in enumerate(doc_ids) if idx not in skip_ids]
        #所有值都被跳过
        if len(docs) == 0:
            return np.zeros(1,top_n)

    #根据之前求出来的值排序
    if doc_ids is not None:
        doc_ids = [x for _, x in sorted(zip(scores, doc_ids), reverse=True)]
    docs_scores = [(x, s) for s, x in sorted(zip(scores, docs), reverse=True)]
    docs = [item[0] for item in docs_scores]
    scores = [item[1] for item in docs_scores]

    top_n_indices = np.argsort(scores)[-top_n:][::-1]
    top_n_blocks = [docs[i] for i in top_n_indices]
    return top_n_blocks

def main(json_file, query_file,rank_fn, top_n=2):
    function_blocks,class_methods = load_function_blocks(json_file)
    query = load_query(query_file)

    top_n_blocks =lexical_ranking(query,function_blocks,rank_fn,top_n)
    # 得到了最相近的 n 个函数
    #print(top_n_blocks)
    for block in top_n_blocks:
        print(f"File: {block.file_path}, Start Line: {block.start_line}, End Line: {block.end_line}")
        #print(get_function_text(block))
        #print("\n")

root_dir=""
top_num = 4 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some files.')
    parser.add_argument('json_file', type=str, help='The JSON file with parsed code.')
    parser.add_argument('query_file', type=str, help='The file containing the query code.')
    parser.add_argument('root_dir', type=str, help='The root directory for code files.')
    parser.add_argument('rank_fn', type=str, choices=['bm25', 'tfidf', 'jaccard_sim', 'openai'], help='The ranking function to use.')

    args = parser.parse_args()
    main(args.json_file, args.query_file,args.rank_fn,top_num)
