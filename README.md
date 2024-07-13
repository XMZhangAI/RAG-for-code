# Retrieved Augement Code Generation: Repo-Domin Tuning
### 本科生的第一个项目
始于 2024.6.10
## 两个 python 文件的介绍
### BlocksCutting.py 使用说明
***概述***  
BlocksCutting.py脚本旨在分析指定目录中的Python代码文件。它提取这些文件中定义的函数和类的信息，并以结构化的JSON格式保存这些信息。该脚本特别适用于代码分析、重构和理解大型代码库。  
***功能***  
函数提取：识别并提取Python文件中的所有函数，捕获函数名称、所属类（如果有）、开始和结束行、调用的函数以及导入的模块等详细信息。  
类方法提取：识别并提取Python文件中的所有类及其方法。  
目录遍历：递归遍历指定目录以查找所有Python文件。  
JSON输出：将提取的信息保存到JSON文件中，以便于使用和分析。  
***组件***  
>类：function_block
一个封装函数详细信息的类，包括：  
name：函数名称。  
belong_class：函数所属的类（如果有）。  
start_line：函数开始的行号。  
end_line：函数结束的行号。  
call_func：该函数调用的函数列表。  
import_repo：包含该函数的文件中导入的模块列表。  
>函数：get_python_files(directory)  
遍历指定目录并返回所有Python文件的路径列表。

>类：FunctionVisitor
一个AST（抽象语法树）访问器类，用于识别和提取Python代码中的函数及其详细信息。

>函数：add_parent_references(node)
为AST中的节点添加父节点引用，以便于向上遍历树。

>函数：parse_functions(file_path)
解析Python文件以提取函数信息及其导入的模块。

>函数：parse_class_methods(file_path)
解析Python文件以提取类及其方法的信息。

>函数：save_to_json(data, output_file)
将提供的数据保存到JSON文件中。

>函数：main(input_directory, output_file)
主函数，负责解析指定目录中的Python文件并将结果保存到JSON文件中。
---
### FunctionsRetrieval.py 文件介绍  
FunctionsRetrieval.py 是一个用于在代码库中根据查询代码片段检索最相关的函数块的 Python 脚本。这个脚本结合了不同的文本相似度计算方法（如 BM25、TF-IDF、Jaccard 相似度和 OpenAI 的嵌入）来对函数块进行排名，并返回最相关的函数块以及它们的相关信息。  
***主要功能***  
加载函数块数据：从指定的 JSON 文件中加载解析后的函数块和类方法信息。  
加载查询代码：从指定的文件中读取查询代码。  
获取函数文本：根据函数块的信息（文件路径、起始行和结束行）获取函数的源代码文本。  
计算相似度：提供了多种相似度计算方法，包括 BM25、TF-IDF、Jaccard 相似度和 OpenAI 嵌入，用于计算查询代码与函数块之间的相似度。  
函数块排序：根据计算的相似度对函数块进行排序，并返回最相关的函数块。  
获取类方法：获取与给定函数块相关的类方法。  
获取调用的函数：获取给定函数块中调用的其他函数，并返回这些函数块。  
交互式输入：支持用户通过交互式输入指定 JSON 文件、查询文件、根目录、排序方法、返回结果数量以及相关参数。  
