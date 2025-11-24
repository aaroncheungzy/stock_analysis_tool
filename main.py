from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from datetime import datetime
import io
import base64
import os
import logging
import warnings
import json
from dotenv import load_dotenv
from utils import search_stocks  # 仅导入股票搜索函数（移除 Redis 相关）

# 配置
load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
warnings.filterwarnings('ignore')

# 设置 matplotlib 后端（无 GUI）
plt.switch_backend('Agg')
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 代理配置（可选）
proxy_url = os.getenv("PROXY_URL")
if proxy_url:
    os.environ["HTTP_PROXY"] = proxy_url
    os.environ["HTTPS_PROXY"] = proxy_url

# 创建 FastAPI 应用
app = FastAPI(title="股票涨跌幅分析 API", version="2.0")

# 允许跨域（生产环境改为你的GitHub Pages域名）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 定义请求参数模型（移除 refresh_cache 字段）
class AnalysisRequest(BaseModel):
    ticker: str          # 股票代码
    target_change: float # 目标涨跌幅
    period: str = "max"  # 数据周期

# 股票搜索接口（保留）
@app.get("/search-stocks", summary="股票搜索（模糊查询）")
async def search_stocks_api(keyword: str = Query(..., min_length=1)):
    """根据关键词搜索股票（支持股票名称、代码模糊查询）"""
    results = search_stocks(keyword)
    return {"success": True, "data": results}

# 分析器类（移除缓存相关逻辑）
class StockVolatilityAnalyzer:
    def __init__(self, ticker, target_change, period='max'):
        self.ticker = ticker
        self.target_change = target_change
        self.period = period
        self.data = None
        self.ticker_info = None
        self.signal_dates = []
        self.future_returns = {}
        self.periods = [1, 7, 15, 30, 60, 180, 360]

    # 数据获取（复用原有逻辑）
    def fetch_data(self):
        try:
            logging.info(f"获取 {self.ticker} 数据（周期：{self.period}）")
            stock = yf.Ticker(self.ticker)
            self.data = stock.history(period=self.period)
            if self.data.empty:
                return False
            self.ticker_info = stock.info if stock.info else {"shortName": self.ticker}
            return True
        except Exception as e:
            logging.error(f"数据获取失败：{str(e)}")
            return False

    # 信号识别（复用原有逻辑）
    def find_signal_dates(self):
        if self.data.empty:
            return False
        self.data['Daily_Change'] = self.data['Close'].pct_change() * 100
        mask = self.data['Daily_Change'] >= self.target_change if self.target_change > 0 else self.data['Daily_Change'] <= self.target_change
        self.signal_dates = self.data[mask].index.tolist()
        return len(self.signal_dates) > 0

    # 收益计算（复用原有逻辑）
    def calculate_future_returns(self):
        if not self.signal_dates:
            return False
        for period in self.periods:
            returns = []
            for date in self.signal_dates:
                idx = self.data.index.get_loc(date)
                if idx + period < len(self.data):
                    ret = (self.data.iloc[idx+period]['Close'] - self.data.iloc[idx]['Close']) / self.data.iloc[idx]['Close'] * 100
                    returns.append(ret)
            self.future_returns[period] = returns
        return True

    # 生成统计结果（复用原有逻辑）
    def generate_statistics(self):
        stats_dict = {}
        for period, returns in self.future_returns.items():
            if returns:
                arr = np.array(returns)
                stats_dict[period] = {
                    'count': len(returns),
                    'mean': round(float(np.mean(arr)), 2),
                    'median': round(float(np.median(arr)), 2),
                    'std': round(float(np.std(arr)), 2),
                    'min': round(float(np.min(arr)), 2),
                    'max': round(float(np.max(arr)), 2),
                    'positive_rate': round(float(np.sum(arr > 0) / len(arr) * 100), 1),
                    'skewness': round(float(stats.skew(arr)), 2),
                    'kurtosis': round(float(stats.kurtosis(arr)), 2)
                }
        return stats_dict

    # 图表生成（复用原有逻辑）
    def plot_to_base64(self, plot_func):
        try:
            buf = io.BytesIO()
            plot_func()
            plt.tight_layout()
            plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
            buf.seek(0)
            base64_str = base64.b64encode(buf.read()).decode('utf-8')
            plt.close()
            return f"data:image/png;base64,{base64_str}"
        except Exception as e:
            logging.error(f"图表生成失败：{str(e)}")
            return None

    def plot_price_with_signals(self):
        def _plot():
            plt.figure(figsize=(14, 7))
            plt.plot(self.data.index, self.data['Close'], 'b-', alpha=0.7, linewidth=1.5, label='收盘价')
            signal_prices = [self.data.loc[d]['Close'] for d in self.signal_dates if d in self.data.index]
            plt.scatter(self.signal_dates, signal_prices, color='red', s=50, alpha=0.7, label=f'涨跌幅{self.target_change}%')
            plt.title(f'{self.ticker} 价格走势与信号点', fontsize=14)
            plt.xlabel('日期'), plt.ylabel('价格'), plt.grid(True, alpha=0.3), plt.legend()
        return self.plot_to_base64(_plot)

    def plot_returns_distribution(self):
        def _plot():
            plt.figure(figsize=(12, 8))
            n_cols = min(3, len(self.future_returns))
            n_rows = (len(self.future_returns) + n_cols - 1) // n_cols
            for i, (period, returns) in enumerate(self.future_returns.items()):
                if returns:
                    ax = plt.subplot(n_rows, n_cols, i+1)
                    ax.hist(returns, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
                    ax.set_title(f'{period}日收益率分布'), ax.set_xlabel('收益率(%)'), ax.set_ylabel('频次')
                    ax.grid(True, alpha=0.3)
            plt.suptitle(f'{self.ticker} 涨跌幅{self.target_change}%后收益率分布', fontsize=14)
        return self.plot_to_base64(_plot)

    def plot_boxplot_comparison(self):
        def _plot():
            plt.figure(figsize=(12, 6))
            data = [self.future_returns[p] for p in sorted(self.future_returns.keys()) if self.future_returns[p]]
            labels = [f'{p}日' for p in sorted(self.future_returns.keys()) if self.future_returns[p]]
            box = plt.boxplot(data, labels=labels, patch_artist=True, boxprops=dict(facecolor='skyblue'))
            means = [np.mean(d) for d in data]
            plt.plot(range(1, len(means)+1), means, 'ro-', label='均值'), plt.legend()
            plt.title(f'{self.ticker} 不同周期收益率分布', fontsize=14)
            plt.xlabel('周期'), plt.ylabel('收益率(%)'), plt.grid(True, alpha=0.3)
            plt.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        return self.plot_to_base64(_plot)

    def plot_cumulative_returns(self):
        def _plot():
            plt.figure(figsize=(12, 6))
            periods = sorted(self.future_returns.keys())
            cum_returns = [self.future_returns[p] for p in periods if self.future_returns[p]]
            avg_returns = [np.mean(r) for r in cum_returns]
            plt.plot(periods, avg_returns, 'b-o', linewidth=2, label='平均累计收益率')
            for p, r in zip(periods, avg_returns):
                plt.text(p, r, f'{r:.2f}%', ha='center', va='bottom', fontsize=9)
            plt.title(f'{self.ticker} 平均累计收益率', fontsize=14)
            plt.xlabel('天数'), plt.ylabel('收益率(%)'), plt.grid(True, alpha=0.3), plt.legend()
            plt.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        return self.plot_to_base64(_plot)

    # 执行完整分析（移除缓存逻辑）
    def run(self):
        if not self.fetch_data():
            raise HTTPException(status_code=400, detail=f"无法获取 {self.ticker} 的数据")
        if not self.find_signal_dates():
            raise HTTPException(status_code=400, detail=f"未找到 {self.ticker} 涨跌幅达到 {self.target_change}% 的日期")
        if not self.calculate_future_returns():
            raise HTTPException(status_code=500, detail="收益率计算失败")
        
        stats_dict = self.generate_statistics()
        return {
            "ticker": self.ticker,
            "target_change": self.target_change,
            "signal_count": len(self.signal_dates),
            "date_range": {
                "start": self.data.index[0].strftime('%Y-%m-%d'),
                "end": self.data.index[-1].strftime('%Y-%m-%d')
            },
            "stats": stats_dict,
            "charts": {
                "price_with_signals": self.plot_price_with_signals(),
                "returns_distribution": self.plot_returns_distribution(),
                "boxplot_comparison": self.plot_boxplot_comparison(),
                "cumulative_returns": self.plot_cumulative_returns()
            },
            "ticker_info": self.ticker_info,
            "generated_at": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

# 核心分析接口（移除缓存相关逻辑）
@app.post("/analyze", summary="股票涨跌幅分析")
async def analyze_stock(request: AnalysisRequest):
    try:
        # 直接执行分析（无缓存逻辑）
        analyzer = StockVolatilityAnalyzer(
            ticker=request.ticker.strip(),
            target_change=request.target_change,
            period=request.period.strip()
        )
        result = analyzer.run()
        
        return {"success": True, "data": result}
    except HTTPException as e:
        return {"success": False, "message": e.detail}
    except Exception as e:
        logging.error(f"分析失败：{str(e)}")
        return {"success": False, "message": f"服务器错误：{str(e)}"}

# 健康检查接口（移除 Redis 状态检查）
@app.get("/health", summary="健康检查")
async def health_check():
    return {
        "status": "healthy",
        "message": "API 服务正常运行"
    }

# 本地运行入口
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
