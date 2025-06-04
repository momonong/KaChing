from tools.fetch import (
    fetch_stock_price,
    fetch_institutional_investors,
    fetch_financial_statement,
    fetch_news,
    fetch_dividend,
    fetch_monthly_revenue,
    fetch_shareholding_distribution,
)


if __name__ == "__main__":
    stock_id = "2330"
    start_date = "2025-01-01"
    end_date = "2025-05-31"

    print("\n📈 股票價格（TaiwanStockPrice）:")
    df_price = fetch_stock_price(stock_id, start_date, end_date)
    print(df_price.head())

    print("\n📊 法人買賣超（TaiwanStockInstitutionalInvestors）:")
    df_investors = fetch_institutional_investors(stock_id, start_date, end_date)
    print(df_investors.head())

    print("\n🧾 財報（TaiwanStockFinancialStatements）:")
    df_financial = fetch_financial_statement(stock_id, start_date, end_date)
    print(df_financial.head())

    print("\n📰 新聞（TaiwanStockNews）:")
    df_news = fetch_news(stock_id, start_date, start_date)
    print(df_news.head())

    print("\n💰 股利（TaiwanStockDividend）:")
    df_dividend = fetch_dividend(stock_id, start_date, end_date)
    print(df_dividend.head())

    print("\n💹 月營收（TaiwanStockMonthRevenue）:")
    df_revenue = fetch_monthly_revenue(stock_id, start_date, end_date)
    print(df_revenue.head())

    print("\n📊 持股分布（TaiwanStockHoldingSharesPer）:")
    df_shareholding = fetch_shareholding_distribution(stock_id, start_date, end_date)
    print(df_shareholding.head())
