import os
import torch
from transformers import RobertaTokenizer, RobertaModel
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import ast
import re

# 初始化 GraphCodeBERT 模型和 Tokenizer
model_name = "microsoft/graphcodebert-base"
tokenizer = RobertaTokenizer.from_pretrained(model_name)
model = RobertaModel.from_pretrained(model_name)

# 读取文件内容并进行预处理
def preprocess_code(code):
    # 移除注释和多余空格
    code = re.sub(r'#.*', '', code)
    code = re.sub(r'\s+', ' ', code).strip()
    return code

def read_code_from_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        code = file.read()
    return preprocess_code(code)

# 提取代码的核心逻辑
def extract_core_logic(code):
    try:
        tree = ast.parse(code)
        core_logic = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.For, ast.While, ast.If)):
                core_logic.append(ast.unparse(node))
        return '\n'.join(core_logic)
    except SyntaxError:
        return code  # 如果代码有语法错误，直接返回原始代码

# 使用 GraphCodeBERT 编码代码片段
def encode_code(code, model, tokenizer):
    inputs = tokenizer(code, return_tensors='pt', max_length=512, truncation=True, padding='max_length')
    with torch.no_grad():
        outputs = model(**inputs)
        last_hidden_state = outputs.last_hidden_state
        vector = last_hidden_state.mean(dim=1).squeeze()
    return vector.numpy()

# 比较目标代码与代码库中的代码相似度
def compare_with_library(target_code_path, library_path, model, tokenizer):
    target_code = read_code_from_file(target_code_path)
    target_code = extract_core_logic(target_code)
    target_vector = encode_code(target_code, model, tokenizer)

    code_library = []
    for i in range(1, 16):  # 读取编号为1到15的代码文件
        file_path = os.path.join(library_path, f"{i}.txt")
        if os.path.exists(file_path):
            code_tokens = read_code_from_file(file_path)
            code_tokens = extract_core_logic(code_tokens)
            code_library.append(code_tokens)
        else:
            print(f"File {i}.txt not found in the code library.")

    similarities = []
    for code_tokens in code_library:
        vector = encode_code(code_tokens, model, tokenizer)
        similarity = cosine_similarity([target_vector], [vector])[0][0]
        similarities.append(similarity)

    return similarities

# 找到相似度最高的三个代码片段
def find_top_three_similarities(similarities):
    top_three_indices = sorted(range(len(similarities)), key=lambda i: similarities[i], reverse=True)[:3]
    return [idx + 1 for idx in top_three_indices]

# 示例代码
if __name__ == "__main__":
    # 目标代码文件路径
    target_code_path = r"attempt\attempt_1.txt"
    # 代码库文件夹路径
    library_path = r"code-ref"

    # 比较目标代码与代码库中的代码相似度
    similarities = compare_with_library(target_code_path, library_path, model, tokenizer)
    print("Similarities with code library:", similarities)
    top_three_indices = find_top_three_similarities(similarities)

    print(f"相似度最高的三个为：{', '.join(map(str, top_three_indices))}")