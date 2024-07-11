# RAG For Code
### 本科生的第一个项目
始于 2024.6.10
## 两个 python 文件的介绍
### README for BlocksCutting.py
Overview
The BlocksCutting.py script is designed to analyze Python code files within a specified directory. It extracts information about functions and classes defined in these files and saves this information in a structured JSON format. This script can be particularly useful for code analysis, refactoring, and understanding large codebases.

Features
Function Extraction: Identifies and extracts all functions in a Python file, capturing details such as the function name, the class it belongs to (if any), the lines where it starts and ends, the functions it calls, and the modules it imports.
Class Method Extraction: Identifies and extracts all classes and their methods in a Python file.
Directory Traversal: Recursively traverses a given directory to find all Python files.
JSON Output: Saves the extracted information into a JSON file for easy consumption and analysis.
Components
Class: function_block
A class that encapsulates the details of a function. It includes:

name: The name of the function.
belong_class: The class to which the function belongs (if any).
start_line: The line number where the function starts.
end_line: The line number where the function ends.
call_func: A list of functions called by this function.
import_repo: A list of modules imported in the file containing this function.
Function: get_python_files(directory)
Traverses the given directory and returns a list of paths to all Python files.

Class: FunctionVisitor
An AST (Abstract Syntax Tree) visitor class that identifies and extracts functions and their details from Python code.

Function: add_parent_references(node)
Adds parent references to nodes in the AST, enabling traversal up the tree.

Function: parse_functions(file_path)
Parses a Python file to extract function information and their imports.

Function: parse_class_methods(file_path)
Parses a Python file to extract class and method information.

Function: save_to_json(data, output_file)
Saves the provided data into a JSON file.

Function: main(input_directory, output_file)
Main function that orchestrates the parsing of Python files in the given directory and saves the results to a JSON file.

Usage
Running the Script
You can run the script directly from the command line. It will prompt you to input the directory containing your Python files:

sh
复制代码
python BlocksCutting.py
You will be prompted to enter the root directory for your code files. The script will then process all Python files in the specified directory and save the extracted information in parsed_code.json.

Example
Input: Suppose you have the following directory structure:
复制代码
my_python_project/
├── module1.py
└── module2.py
Running the Script:
sh
复制代码
python BlocksCutting.py
Please enter the root directory for code files: my_python_project
Output: The script will generate a parsed_code.json file with the extracted information:
json
复制代码
{
    "my_python_project/module1.py": {
        "Functions": [
            {
                "name": "func1",
                "class": null,
                "lineno": 10,
                "end_lineno": 20,
                "calls": ["func2"],
                "import": ["os", "sys"]
            }
        ],
        "Classes": [
            {
                "class_name": "MyClass",
                "methods": ["method1", "method2"]
            }
        ]
    },
    "my_python_project/module2.py": {
        "Functions": [],
        "Classes": []
    }
}
Dependencies
Python 3.x
Notes
The script assumes UTF-8 encoding for Python files.
Currently, the script does not handle nested functions.
Imported functions and methods are included in the calls list as fully qualified names (e.g., module.class.method).
License
This project is licensed under the MIT License.
