import requests
import json

BASE_URL = "http://localhost:8000/api/v1"

def print_response(response):
    print(json.dumps(response.json(), indent=2))

def example_get_schema():
    """获取领域模式示例"""
    print("\n1. 获取经济学领域的因果模式")
    response = requests.get(f"{BASE_URL}/schema/economics")
    print_response(response)

def example_evaluate_causal_reasoning():
    """因果推理评估示例"""
    print("\n2. 评估经济场景中的因果关系")
    data = {
        "domain": "economics",
        "query_variable": "C2",
        "c1_state": 1,
        "e_state": 1,
        "model": "claude"
    }
    response = requests.post(f"{BASE_URL}/evaluate_causal_reasoning", json=data)
    print_response(response)

def example_compare_models():
    """比较不同模型的推理结果"""
    print("\n3. 比较不同模型对同一场景的推理")
    models = ["gpt-4", "claude", "gemini", "gpt-3.5"]
    data = {
        "domain": "meteorology",
        "query_variable": "E",
        "c1_state": 1,
        "c2_state": 1
    }
    
    for model in models:
        data["model"] = model
        response = requests.post(f"{BASE_URL}/evaluate_causal_reasoning", json=data)
        print(f"\n{model.upper()} 的推理结果：")
        print_response(response)

if __name__ == "__main__":
    print("运行 API 使用示例...")
    example_get_schema()
    example_evaluate_causal_reasoning()
    example_compare_models()
