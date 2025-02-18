import pytest
from fastapi.testclient import TestClient
from server import app, ModelType, SCENARIOS

client = TestClient(app)

# 测试基本 API 功能
@pytest.mark.parametrize("domain", list(SCENARIOS.keys()))
def test_get_domain_schema(domain):
    response = client.get(f"/api/v1/schema/{domain}")
    assert response.status_code == 200
    data = response.json()
    assert "schema" in data
    assert "variables" in data
    assert "description" in data["variables"]
    assert all(key in data["variables"] for key in ["C1", "C2", "E"])

# 测试因果推理功能
@pytest.mark.parametrize("model", [m.value for m in ModelType])
@pytest.mark.parametrize("domain", list(SCENARIOS.keys()))
@pytest.mark.parametrize("query_variable", ["C1", "C2", "E"])
def test_evaluate_causal_reasoning(model, domain, query_variable):
    request_data = {
        "domain": domain,
        "query_variable": query_variable,
        "c1_state": 1,
        "c2_state": 1,
        "e_state": 1,
        "model": model
    }
    response = client.post("/api/v1/evaluate_causal_reasoning", json=request_data)
    assert response.status_code == 200
    data = response.json()
    assert "probability" in data
    assert 0 <= data["probability"] <= 1
    assert "explanation" in data
    assert model in data["explanation"]

# 测试解释效应
@pytest.mark.parametrize("model", [m.value for m in ModelType])
def test_explaining_away_effect(model):
    # 测试当E=1时的解释效应
    request_data_1 = {
        "domain": "economics",
        "query_variable": "C1",
        "e_state": 1,
        "c2_state": 1,
        "model": model
    }
    request_data_2 = {
        "domain": "economics",
        "query_variable": "C1",
        "e_state": 1,
        "c2_state": 0,
        "model": model
    }
    
    response_1 = client.post("/api/v1/evaluate_causal_reasoning", json=request_data_1)
    response_2 = client.post("/api/v1/evaluate_causal_reasoning", json=request_data_2)
    
    prob_1 = response_1.json()["probability"]
    prob_2 = response_2.json()["probability"]
    
    # 当另一个原因存在时，概率应该更低
    assert prob_1 < prob_2

# 测试边界条件
def test_invalid_inputs():
    # 测试无效的域
    response = client.get("/api/v1/schema/invalid_domain")
    assert response.status_code == 200
    assert "error" in response.json()
    
    # 测试无效的查询变量
    request_data = {
        "domain": "economics",
        "query_variable": "invalid_var",
        "c1_state": 1,
        "e_state": 1
    }
    response = client.post("/api/v1/evaluate_causal_reasoning", json=request_data)
    assert response.status_code == 200
    assert "error" in response.json()
    
    # 测试无效的状态值
    request_data = {
        "domain": "economics",
        "query_variable": "C1",
        "c1_state": 2,  # 应该只能是 0 或 1
        "e_state": 1
    }
    response = client.post("/api/v1/evaluate_causal_reasoning", json=request_data)
    assert response.status_code == 422  # FastAPI 的验证错误

# 测试模型特征
@pytest.mark.parametrize("model1,model2", [
    (ModelType.GPT4, ModelType.GEMINI),  # GPT4 应该更接近规范推理
    (ModelType.CLAUDE, ModelType.GPT35)  # Claude 也应该更接近规范推理
])
def test_model_characteristics(model1, model2):
    request_data = {
        "domain": "healthcare",
        "query_variable": "E",
        "c1_state": 1,
        "c2_state": 1
    }
    
    # 收集多个样本
    samples = 10
    probs1 = []
    probs2 = []
    
    for _ in range(samples):
        request_data["model"] = model1.value
        response1 = client.post("/api/v1/evaluate_causal_reasoning", json=request_data)
        probs1.append(response1.json()["probability"])
        
        request_data["model"] = model2.value
        response2 = client.post("/api/v1/evaluate_causal_reasoning", json=request_data)
        probs2.append(response2.json()["probability"])
    
    # 计算平均值和标准差
    avg1 = sum(probs1) / len(probs1)
    avg2 = sum(probs2) / len(probs2)
    
    # 验证模型特征
    if model1 in [ModelType.GPT4, ModelType.CLAUDE]:
        assert avg1 > avg2  # 更接近规范推理的模型应该有更高的概率
