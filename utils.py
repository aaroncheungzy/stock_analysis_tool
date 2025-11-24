import redis
import os
import requests
from dotenv import load_dotenv
import logging

load_dotenv()
logger = logging.getLogger(__name__)

# Redis连接初始化
redis_client = None
if os.getenv("REDIS_URL"):
    try:
        redis_client = redis.from_url(os.getenv("REDIS_URL"), decode_responses=False)
        # 测试连接
        redis_client.ping()
        logger.info("Redis缓存连接成功")
    except Exception as e:
        logger.error(f"Redis连接失败：{str(e)}")
        redis_client = None

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

# 缓存工具函数
def get_cache(key: str):
    """从Redis获取缓存"""
    if not redis_client:
        return None
    try:
        return redis_client.get(key)
    except Exception as e:
        logger.error(f"获取缓存失败：{str(e)}")
        return None

def set_cache(key: str, value: bytes, expire_seconds: int = 86400):
    """设置Redis缓存（默认过期1天）"""
    if not redis_client:
        return
    try:
        redis_client.setex(key, expire_seconds, value)
    except Exception as e:
        logger.error(f"设置缓存失败：{str(e)}")

def delete_cache(key: str):
    """删除Redis缓存"""
    if not redis_client:
        return
    try:
        redis_client.delete(key)
    except Exception as e:
        logger.error(f"删除缓存失败：{str(e)}")
