from typing import Dict, Optional, List
from fastapi import FastAPI, APIRouter, Query
from pydantic import BaseModel
from enum import Enum
import uvicorn
import random

app = FastAPI(title="Causal Reasoning MCP Server")
router = APIRouter()

class ModelType(str, Enum):
    GPT4 = "gpt-4"
    CLAUDE = "claude"
    GEMINI = "gemini"
    GPT35 = "gpt-3.5"

class CausalRequest(BaseModel):
    domain: str
    query_variable: str
    c1_state: Optional[int] = None
    c2_state: Optional[int] = None
    e_state: Optional[int] = None
    model: Optional[ModelType] = ModelType.GPT4

def simulate_model_response(domain: str, query_var: str, states: Dict[str, Optional[int]], model: ModelType) -> float:
    """模拟不同模型的因果推理响应"""
    base_probability = random.random() * 0.5 + 0.25  # 基础概率在 0.25-0.75 之间
    
    # 模型特定的偏差
    model_bias = {
        ModelType.GPT4: 0.1,     # 更接近规范推理
        ModelType.CLAUDE: 0.05,   # 最接近规范推理
        ModelType.GEMINI: -0.15,  # 较大偏差
        ModelType.GPT35: -0.1     # 中等偏差
    }
    
    # 解释效应：当效果存在时降低其他原因的概率
    if states.get('E') == 1 and query_var in ['C1', 'C2']:
        other_cause = 'C2' if query_var == 'C1' else 'C1'
        if states.get(other_cause) == 1:
            base_probability *= 0.7  # 解释效应
    
    return min(1.0, max(0.0, base_probability + model_bias[model]))

# Define domain-specific scenarios
SCENARIOS = {
    "economics": {
        "C1": "Economic recession",
        "C2": "Market competition",
        "E": "Business failure",
        "description": "研究经济衰退和市场竞争如何影响企业生存"
    },
    "meteorology": {
        "C1": "Low pressure system",
        "C2": "High humidity",
        "E": "Rainfall",
        "description": "研究低气压系统和高湿度如何影响降雨"
    },
    "sociology": {
        "C1": "Social isolation",
        "C2": "Financial stress",
        "E": "Depression",
        "description": "研究社交隔离和经济压力如何影响抑郁症"
    },
    "healthcare": {
        "C1": "Genetic predisposition",
        "C2": "Environmental factors",
        "E": "Disease onset",
        "description": "研究遗传倾向和环境因素如何影响疾病发生"
    },
    "education": {
        "C1": "Study habits",
        "C2": "Teacher quality",
        "E": "Academic performance",
        "description": "研究学习习惯和教师质量如何影响学术表现"
    },
    "technology": {
        "C1": "Technical innovation",
        "C2": "Market demand",
        "E": "Product success",
        "description": "研究技术创新和市场需求如何影响产品成功"
    }
}

@router.get("/schema/{domain}")
async def get_domain_schema(domain: str) -> Dict:
    """Get the causal schema for a specific domain"""
    if domain not in SCENARIOS:
        return {"error": f"Domain {domain} not found"}
    
    scenario = SCENARIOS[domain]
    schema = f"Causal Schema for {domain}:\n" + \
             f"- Cause 1 (C1): {scenario['C1']}\n" + \
             f"- Cause 2 (C2): {scenario['C2']}\n" + \
             f"- Effect (E): {scenario['E']}"
    
    return {
        "schema": schema,
        "domain": domain,
        "variables": scenario
    }

@router.post("/evaluate_causal_reasoning")
async def evaluate_causal_reasoning(request: CausalRequest) -> Dict:
    """Evaluate causal reasoning for a collider graph scenario
    
    Args:
        domain: Domain of the scenario (economics, meteorology, or sociology)
        query_variable: Variable to query (C1, C2, or E)
        c1_state: State of cause 1 (0 or 1, optional)
        c2_state: State of cause 2 (0 or 1, optional)
        e_state: State of effect (0 or 1, optional)
    """
    if request.domain not in SCENARIOS:
        return {"error": f"Domain {request.domain} not found"}
    
    if request.query_variable not in ["C1", "C2", "E"]:
        return {"error": f"Invalid query variable {request.query_variable}"}
    
    scenario = SCENARIOS[request.domain]
    
    # Build prompt based on scenario and states
    prompt = f"Domain: {request.domain}\n"
    prompt += f"Causal scenario:\n"
    prompt += f"- {scenario['C1']} (C1) {'is present' if request.c1_state == 1 else 'is absent' if request.c1_state == 0 else 'state unknown'}\n"
    prompt += f"- {scenario['C2']} (C2) {'is present' if request.c2_state == 1 else 'is absent' if request.c2_state == 0 else 'state unknown'}\n"
    prompt += f"- {scenario['E']} (E) {'is present' if request.e_state == 1 else 'is absent' if request.e_state == 0 else 'state unknown'}\n"
    prompt += f"\nQuery: What is the likelihood of {scenario[request.query_variable]} being present?"

    # 获取模型响应
    states = {
        "C1": request.c1_state,
        "C2": request.c2_state,
        "E": request.e_state
    }
    probability = simulate_model_response(
        domain=request.domain,
        query_var=request.query_variable,
        states=states,
        model=request.model
    )
    
    return {
        "prompt": prompt,
        "domain": request.domain,
        "scenario": scenario,
        "states": states,
        "query_variable": request.query_variable,
        "model": request.model,
        "probability": probability,
        "explanation": f"Based on the {request.model} model's analysis, "
                      f"the likelihood of {scenario[request.query_variable]} being present is {probability:.2%}."
    }

# Register the router
app.include_router(router, prefix="/api/v1")

# Add API documentation
app.title = "Causal Reasoning API"
app.description = """
## Causal Reasoning Server

This API implements causal reasoning tasks using collider graphs across multiple domains:
- Economics: Economic recession → Business failure ← Market competition
- Meteorology: Low pressure system → Rainfall ← High humidity
- Sociology: Social isolation → Depression ← Financial stress

Based on research paper arXiv-2502.10215v1.
"""
app.version = "1.0.0"

if __name__ == "__main__":
    print("Starting Causal Reasoning server...")
    print("Available endpoints:")
    print("- POST /api/v1/evaluate_causal_reasoning")
    print("\nAPI documentation:")
    print("- Swagger UI: http://localhost:8000/docs")
    print("- ReDoc: http://localhost:8000/redoc")
    print("\nServer will be available at http://localhost:8000")
    
    # Run with uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, log_level="info", reload=True)
