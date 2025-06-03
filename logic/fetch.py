import pandas as pd
import requests
import os
from dotenv import load_dotenv

load_dotenv()


FINMIND_BASE_URL = os.getenv("FINMIND_BASE_URL")
FINMIND_TOKEN = os.getenv("FINMIND_TOKEN")

print(FINMIND_BASE_URL)
print(FINMIND_TOKEN)


def get_dataset(
    dataset: str, data_id: str, start_date: str, end_date: str
) -> pd.DataFrame:
    """
    通用方法：從 FinMind API 取得指定 dataset 的資料
    """
    headers = {"Authorization": f"Bearer {FINMIND_TOKEN}"}
    params = {
        "dataset": dataset,
        "data_id": data_id,
        "start_date": start_date,
        "end_date": end_date,
    }
    resp = requests.get(FINMIND_BASE_URL, headers=headers, params=params)
    if resp.status_code != 200:
        print(f"❌ API 回傳錯誤: {resp.status_code} - {resp.text}")
        resp.raise_for_status()
    data = resp.json()
    if data.get("data"):
        return pd.DataFrame(data["data"])
    return pd.DataFrame()


def fetch_stock_price(stock_id: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    擷取股票日收盤價等基本價格資料
    """
    return get_dataset("TaiwanStockPrice", stock_id, start_date, end_date)


def fetch_institutional_investors(
    stock_id: str, start_date: str, end_date: str
) -> pd.DataFrame:
    """
    擷取法人買賣超資料（外資、投信、自營商）
    """
    return get_dataset(
        "TaiwanStockInstitutionalInvestorsBuySell", stock_id, start_date, end_date
    )


def fetch_financial_statement(
    stock_id: str, start_date: str, end_date: str
) -> pd.DataFrame:
    """
    擷取財報資訊（營收、EPS、資產負債）
    """
    return get_dataset("TaiwanStockFinancialStatements", stock_id, start_date, end_date)


def fetch_news(stock_id: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    擷取股票相關新聞標題與內容
    """
    return get_dataset("TaiwanStockNews", stock_id, start_date, end_date)


def fetch_dividend(stock_id: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    擷取股利政策與發放紀錄
    """
    return get_dataset("TaiwanStockDividend", stock_id, start_date, end_date)


def fetch_monthly_revenue(
    stock_id: str, start_date: str, end_date: str
) -> pd.DataFrame:
    """
    擷取月營收資料，用於觀察營收成長趨勢
    """
    return get_dataset("TaiwanStockMonthRevenue", stock_id, start_date, end_date)


def fetch_shareholding_distribution(*args, **kwargs):
    print("🔒 持股分佈需要 FinMind 贊助等級，目前已略過")
    return pd.DataFrame()
