from fastapi import FastAPI, Query
import requests
import os
import warnings

warnings.filterwarnings('ignore')
app = FastAPI()

# 从环境变量获取 Alpha Vantage API Key（GitHub 仓库配置）
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY", "")

@app.get("/")
async def search_stocks(keyword: str = Query(..., min_length=1)):
    """股票搜索接口（基于 Alpha Vantage）"""
    if not ALPHA_VANTAGE_API_KEY:
        return {"success": False, "message": "未配置 Alpha Vantage API Key"}
    
    url = "https://www.alphavantage.co/query"
    params = {
        "function": "SYMBOL_SEARCH",
        "keywords": keyword,
        "apikey": ALPHA_VANTAGE_API_KEY,
        "outputsize": "compact"
    }
    
    try:
        response = requests.get(url, params=params, timeout=8)  # 缩短超时时间
        response.raise_for_status()
        data = response.json()
        
        if "bestMatches" not in data or len(data["bestMatches"]) == 0:
            return {"success": True, "data": []}
        
        # 格式化结果
        results = []
        for match in data["bestMatches"][:8]:  # 减少返回数量加快速度
            results.append({
                "symbol": match["1. symbol"],
                "name": match["2. name"],
                "type": match["3. type"],
                "region": match["4. region"],
                "currency": match["8. currency"],
                "matchScore": match["9. matchScore"]
            })
        return {"success": True, "data": results}
    except Exception as e:
        return {"success": False, "message": f"搜索失败：{str(e)[:50]}"}

# 健康检查接口
@app.get("/health")
async def health_check():
    return {"status": "healthy", "message": "搜索接口正常运行"}
