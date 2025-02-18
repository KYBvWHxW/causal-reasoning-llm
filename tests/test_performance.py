import pytest
import time
import statistics
from concurrent.futures import ThreadPoolExecutor
from fastapi.testclient import TestClient
from causal_reasoning_llm.server import app, SCENARIOS, ModelType

client = TestClient(app)

def measure_response_time(request_data):
    """测量单个请求的响应时间"""
    start_time = time.time()
    response = client.post("/api/v1/evaluate_causal_reasoning", json=request_data)
    end_time = time.time()
    assert response.status_code == 200
    return end_time - start_time

def test_response_time():
    """测试响应时间"""
    request_data = {
        "domain": "economics",
        "query_variable": "C2",
        "c1_state": 1,
        "e_state": 1,
        "model": "gpt-4"
    }
    
    # 收集100个样本
    response_times = []
    for _ in range(100):
        response_time = measure_response_time(request_data)
        response_times.append(response_time)
    
    avg_time = statistics.mean(response_times)
    p95_time = statistics.quantiles(response_times, n=20)[18]  # 95th percentile
    
    print(f"\n响应时间统计:")
    print(f"平均响应时间: {avg_time:.3f} 秒")
    print(f"95th 百分位响应时间: {p95_time:.3f} 秒")
    
    # 验证性能要求
    assert avg_time < 1.0, f"平均响应时间 ({avg_time:.3f}s) 超过阈值 (1.0s)"
    assert p95_time < 2.0, f"95th 百分位响应时间 ({p95_time:.3f}s) 超过阈值 (2.0s)"

def test_concurrent_requests():
    """测试并发请求处理"""
    request_data = {
        "domain": "healthcare",
        "query_variable": "E",
        "c1_state": 1,
        "c2_state": 1,
        "model": "claude"
    }
    
    # 使用线程池模拟并发请求
    num_concurrent = 10
    num_requests = 50
    
    with ThreadPoolExecutor(max_workers=num_concurrent) as executor:
        start_time = time.time()
        futures = [
            executor.submit(measure_response_time, request_data)
            for _ in range(num_requests)
        ]
        response_times = [future.result() for future in futures]
        total_time = time.time() - start_time
    
    avg_time = statistics.mean(response_times)
    throughput = num_requests / total_time
    
    print(f"\n并发性能统计:")
    print(f"总请求数: {num_requests}")
    print(f"并发级别: {num_concurrent}")
    print(f"总执行时间: {total_time:.3f} 秒")
    print(f"平均响应时间: {avg_time:.3f} 秒")
    print(f"吞吐量: {throughput:.1f} 请求/秒")
    
    # 验证性能要求
    assert throughput > 2, f"吞吐量 ({throughput:.1f} 请求/秒) 低于阈值 (2 请求/秒)"

def test_memory_usage():
    """测试内存使用"""
    import psutil
    import os
    
    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    # 执行一系列请求
    for domain in SCENARIOS.keys():
        for model in ModelType:
            request_data = {
                "domain": domain,
                "query_variable": "E",
                "c1_state": 1,
                "c2_state": 1,
                "model": model.value
            }
            response = client.post("/api/v1/evaluate_causal_reasoning", json=request_data)
            assert response.status_code == 200
    
    final_memory = process.memory_info().rss / 1024 / 1024  # MB
    memory_increase = final_memory - initial_memory
    
    print(f"\n内存使用统计:")
    print(f"初始内存: {initial_memory:.1f} MB")
    print(f"最终内存: {final_memory:.1f} MB")
    print(f"内存增长: {memory_increase:.1f} MB")
    
    # 验证内存使用要求
    assert memory_increase < 100, f"内存增长 ({memory_increase:.1f} MB) 超过阈值 (100 MB)"

def test_error_handling():
    """测试错误处理性能"""
    invalid_requests = [
        # 无效的域
        {
            "domain": "invalid_domain",
            "query_variable": "C1",
            "c1_state": 1
        },
        # 无效的查询变量
        {
            "domain": "economics",
            "query_variable": "invalid_var",
            "c1_state": 1
        },
        # 无效的状态值
        {
            "domain": "economics",
            "query_variable": "C1",
            "c1_state": 2
        }
    ]
    
    response_times = []
    for request_data in invalid_requests:
        start_time = time.time()
        response = client.post("/api/v1/evaluate_causal_reasoning", json=request_data)
        end_time = time.time()
        response_times.append(end_time - start_time)
        assert response.status_code in [200, 422]  # 验证错误响应
    
    avg_error_time = statistics.mean(response_times)
    
    print(f"\n错误处理性能:")
    print(f"平均错误响应时间: {avg_error_time:.3f} 秒")
    
    # 验证错误处理性能要求
    assert avg_error_time < 0.05, f"错误处理平均时间 ({avg_error_time:.3f}s) 超过阈值 (0.05s)"
