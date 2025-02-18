import requests
import json
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict

BASE_URL = "http://localhost:8000/api/v1"
MODELS = ["gpt-4", "claude", "gemini", "gpt-3.5"]

def collect_samples(domain: str, query_variable: str, conditions: Dict[str, int], samples: int = 50) -> Dict[str, List[float]]:
    """收集不同模型的推理样本"""
    results = {model: [] for model in MODELS}
    
    for model in MODELS:
        request_data = {
            "domain": domain,
            "query_variable": query_variable,
            "model": model,
            **conditions
        }
        
        for _ in range(samples):
            response = requests.post(f"{BASE_URL}/evaluate_causal_reasoning", json=request_data)
            if response.status_code == 200:
                results[model].append(response.json()["probability"])
    
    return results

def plot_model_comparison(results: Dict[str, List[float]], title: str):
    """绘制模型比较图"""
    plt.figure(figsize=(10, 6))
    
    # 绘制箱线图
    plt.boxplot([results[model] for model in MODELS], labels=MODELS)
    
    # 添加标题和标签
    plt.title(title)
    plt.ylabel("Probability")
    plt.xlabel("Model")
    
    # 旋转x轴标签
    plt.xticks(rotation=45)
    
    # 添加网格
    plt.grid(True, alpha=0.3)
    
    return plt

def analyze_explaining_away(domain: str = "economics", samples: int = 50):
    """分析解释效应"""
    # 收集数据：当另一个原因存在/不存在时的概率
    results_with_other = collect_samples(
        domain=domain,
        query_variable="C1",
        conditions={"e_state": 1, "c2_state": 1},
        samples=samples
    )
    
    results_without_other = collect_samples(
        domain=domain,
        query_variable="C1",
        conditions={"e_state": 1, "c2_state": 0},
        samples=samples
    )
    
    # 计算解释效应强度
    effect_strength = {}
    for model in MODELS:
        avg_with = np.mean(results_with_other[model])
        avg_without = np.mean(results_without_other[model])
        effect_strength[model] = avg_without - avg_with
    
    # 绘制解释效应强度比较
    plt.figure(figsize=(10, 6))
    models = list(effect_strength.keys())
    strengths = list(effect_strength.values())
    
    plt.bar(models, strengths)
    plt.title(f"Explaining Away Effect Strength ({domain})")
    plt.ylabel("Effect Strength (probability difference)")
    plt.xlabel("Model")
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    return plt

def main():
    # 1. 基本因果推理比较
    print("收集因果推理样本...")
    results = collect_samples(
        domain="healthcare",
        query_variable="E",
        conditions={"c1_state": 1, "c2_state": 1}
    )
    
    plot = plot_model_comparison(
        results,
        "Model Comparison: Disease Onset Probability\nGiven Genetic Predisposition and Environmental Factors"
    )
    plot.savefig("model_comparison.png", bbox_inches="tight")
    print("已保存模型比较图到 model_comparison.png")
    
    # 2. 解释效应分析
    print("\n分析解释效应...")
    plot = analyze_explaining_away(domain="economics")
    plot.savefig("explaining_away_effect.png", bbox_inches="tight")
    print("已保存解释效应分析图到 explaining_away_effect.png")
    
    # 3. 输出详细结果
    print("\n详细结果:")
    for model in MODELS:
        avg = np.mean(results[model])
        std = np.std(results[model])
        print(f"{model}:")
        print(f"  平均概率: {avg:.3f}")
        print(f"  标准差: {std:.3f}")

if __name__ == "__main__":
    main()
