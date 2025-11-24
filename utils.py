import os
import requests
from dotenv import load_dotenv
import logging

load_dotenv()
logger = logging.getLogger(__name__)

# 移除 Redis 相关代码

# Alpha Vantage 股票搜索（支持模糊查询）
def search_stocks(keyword: str) -> list:
    """根据关键词搜索股票（美股为主，支持部分全球市场）"""
    api_key = os.getenv("ALPHA_VANTAGE_API_KEY")
    if not api_key:
        logger.error("未配置Alpha Vantage API Key")
        return []
    
    url = f"https://www.alphavantage.co/query"
    params = {
        "function": "SYMBOL_SEARCH",
        "keywords": keyword,
        "apikey": api_key,
        "outputsize": "compact"
    }
    
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        if "bestMatches" not in data:
            return []
        
        # 格式化结果（保留关键信息）
        results = []
        for match in data["bestMatches"]:
            results.append({
                "symbol": match["1. symbol"],
                "name": match["2. name"],
                "type": match["3. type"],
                "region": match["4. region"],
                "marketOpen": match["5. marketOpen"],
                "marketClose": match["6. marketClose"],
                "timezone": match["7. timezone"],
                "currency": match["8. currency"],
                "matchScore": match["9. matchScore"]
            })
        return results[:10]  # 最多返回10条结果
    except Exception as e:
        logger.error(f"股票搜索失败：{str(e)}")
        return []
