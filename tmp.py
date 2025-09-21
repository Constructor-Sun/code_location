import ast
import os

def extract_calls(file_path, target_func=None):
    """提取Python文件中的函数调用"""
    with open(file_path, 'r') as f:
        tree = ast.parse(f.read())
    
    calls = []
    
    class CallVisitor(ast.NodeVisitor):
        def visit_Call(self, node):
            if isinstance(node.func, ast.Name):
                func_name = node.func.id
                if target_func is None or func_name == target_func:
                    calls.append({
                        'function': func_name,
                        'line': node.lineno,
                        'caller': self.current_function
                    })
            self.generic_visit(node)
    
    visitor = CallVisitor()
    visitor.current_function = None
    
    # 先遍历找到所有函数定义
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            visitor.current_function = node.name
            visitor.visit(node)
    
    return calls

# 使用示例
calls = extract_calls('swe-bench-lite/pytest-dev__pytest-7220/src/_pytest/_code/code.py', 'filter_traceback')
for call in calls:
    print(f"Function {call['caller']} calls {call['function']} at line {call['line']}")