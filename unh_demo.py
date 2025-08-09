#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
UNH股票未来2年走势预测演示
UnitedHealth Group (UNH) Stock Price Prediction Demo
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def fetch_unh_data():
    """获取UNH股票数据"""
    print("正在获取UNH股票数据...")
    ticker = yf.Ticker("UNH")
    data = ticker.history(period="5y")
    print(f"成功获取 {len(data)} 条历史数据")
    return data

def calculate_indicators(data):
    """计算技术指标"""
    # 移动平均线
    data['MA20'] = data['Close'].rolling(window=20).mean()
    data['MA50'] = data['Close'].rolling(window=50).mean()
    data['MA200'] = data['Close'].rolling(window=200).mean()
    
    # RSI
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))
    
    return data

def predict_future_prices(data, days=730):
    """预测未来价格"""
    current_price = data['Close'].iloc[-1]
    predictions = {}
    
    # 方法1: 线性回归
    x = np.arange(len(data))
    y = data['Close'].values
    coeffs = np.polyfit(x, y, 1)
    future_x = np.arange(len(data), len(data) + days)
    linear_pred = np.polyval(coeffs, future_x)
    linear_pred = np.clip(linear_pred, current_price * 0.3, current_price * 5)
    predictions['Linear'] = linear_pred
    
    # 方法2: 蒙特卡洛模拟
    returns = data['Close'].pct_change().dropna()
    mean_return = returns.mean()
    std_return = min(returns.std(), 0.03)  # 限制波动率
    
    mc_predictions = []
    for _ in range(50):  # 50次模拟
        sim_prices = [current_price]
        for _ in range(days):
            daily_return = np.random.normal(mean_return, std_return)
            daily_return = np.clip(daily_return, -0.05, 0.05)  # 限制日变化
            new_price = sim_prices[-1] * (1 + daily_return)
            new_price = max(new_price, current_price * 0.2)
            sim_prices.append(new_price)
        mc_predictions.append(sim_prices[1:])
    
    predictions['Monte_Carlo'] = np.array(mc_predictions)
    
    # 方法3: 基于历史趋势
    recent_trend = (data['Close'].iloc[-1] - data['Close'].iloc[-60]) / 60
    trend_predictions = []
    for i in range(days):
        new_price = current_price + (recent_trend * (i + 1))
        new_price = max(new_price, current_price * 0.5)
        trend_predictions.append(new_price)
    
    predictions['Trend_Based'] = trend_predictions
    
    return predictions

def generate_summary_report(data, predictions):
    """生成总结报告"""
    current_price = data['Close'].iloc[-1]
    
    print("\n" + "="*60)
    print("UNH股票未来2年走势预测总结")
    print("="*60)
    
    print(f"\n📊 当前市场状况:")
    print(f"当前价格: ${current_price:.2f}")
    print(f"52周最高: ${data['High'].max():.2f}")
    print(f"52周最低: ${data['Low'].min():.2f}")
    print(f"历史平均: ${data['Close'].mean():.2f}")
    
    print(f"\n📈 技术指标:")
    print(f"RSI: {data['RSI'].iloc[-1]:.1f}")
    print(f"MA20: ${data['MA20'].iloc[-1]:.2f}")
    print(f"MA50: ${data['MA50'].iloc[-1]:.2f}")
    print(f"MA200: ${data['MA200'].iloc[-1]:.2f}")
    
    print(f"\n🔮 预测结果:")
    final_prices = []
    
    for method, pred in predictions.items():
        if method == 'Monte_Carlo':
            mean_pred = np.mean(pred, axis=0)
            final_price = mean_pred[-1]
            confidence_interval = np.percentile(pred[:, -1], [25, 75])
            print(f"{method:15}: ${final_price:.2f}")
            print(f"{'':15}  置信区间: ${confidence_interval[0]:.2f} - ${confidence_interval[1]:.2f}")
        else:
            final_price = pred[-1]
            print(f"{method:15}: ${final_price:.2f}")
        
        final_prices.append(final_price)
    
    avg_final_price = np.mean(final_prices)
    avg_change_pct = ((avg_final_price - current_price) / current_price) * 100
    
    print(f"\n📋 综合预测:")
    print(f"平均目标价: ${avg_final_price:.2f}")
    print(f"预期涨幅: {avg_change_pct:+.2f}%")
    
    # 风险评估
    if avg_change_pct > 20:
        risk_level = "🟢 低风险 - 强烈看涨"
    elif avg_change_pct > 10:
        risk_level = "🟡 中低风险 - 看涨"
    elif avg_change_pct > 0:
        risk_level = "🟠 中等风险 - 温和看涨"
    elif avg_change_pct > -10:
        risk_level = "🟡 中高风险 - 看跌"
    else:
        risk_level = "🔴 高风险 - 强烈看跌"
    
    print(f"\n⚠️ 风险评估:")
    print(f"风险等级: {risk_level}")
    
    # 投资建议
    print(f"\n💡 投资建议:")
    if avg_change_pct > 15:
        print("✅ 建议买入或持有")
        print("📈 关注技术指标确认信号")
        print("💰 可考虑分批建仓")
    elif avg_change_pct > 5:
        print("✅ 建议谨慎买入")
        print("📊 分批建仓，设置止损")
        print("📋 关注基本面变化")
    elif avg_change_pct > -5:
        print("⏸️ 建议观望")
        print("👀 等待更好的入场机会")
        print("📰 关注市场动态")
    else:
        print("❌ 建议卖出或观望")
        print("🛡️ 等待价格企稳后再考虑")
        print("⚠️ 关注风险控制")

def plot_predictions(data, predictions):
    """绘制预测图表"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # 历史价格和预测
    ax1.plot(data.index, data['Close'], label='历史价格', linewidth=2, color='blue')
    ax1.plot(data.index, data['MA20'], label='MA20', alpha=0.7, color='orange')
    ax1.plot(data.index, data['MA50'], label='MA50', alpha=0.7, color='red')
    
    # 预测线
    future_dates = pd.date_range(start=data.index[-1], periods=731, freq='D')[1:]
    colors = ['green', 'purple', 'brown']
    
    for i, (method, pred) in enumerate(predictions.items()):
        if method == 'Monte_Carlo':
            mean_pred = np.mean(pred, axis=0)
            ax1.plot(future_dates[:len(mean_pred)], mean_pred, 
                    label=f'{method}预测', color=colors[i], linestyle='--', linewidth=2)
        else:
            ax1.plot(future_dates[:len(pred)], pred, 
                    label=f'{method}预测', color=colors[i], linestyle='--', linewidth=2)
    
    ax1.set_title('UNH股票价格走势与未来2年预测', fontsize=14, fontweight='bold')
    ax1.set_ylabel('价格 ($)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 蒙特卡洛模拟分布
    if 'Monte_Carlo' in predictions:
        final_prices = predictions['Monte_Carlo'][:, -1]
        ax2.hist(final_prices, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        ax2.axvline(np.mean(final_prices), color='red', linestyle='--', 
                   label=f'平均: ${np.mean(final_prices):.2f}')
        ax2.set_title('蒙特卡洛模拟结果分布', fontsize=14, fontweight='bold')
        ax2.set_xlabel('预测价格 ($)')
        ax2.set_ylabel('频次')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def main():
    """主函数"""
    print("UNH股票未来2年走势预测演示")
    print("="*50)
    
    # 获取数据
    data = fetch_unh_data()
    
    # 计算指标
    data = calculate_indicators(data)
    
    # 预测价格
    predictions = predict_future_prices(data)
    
    # 生成报告
    generate_summary_report(data, predictions)
    
    # 绘制图表
    plot_predictions(data, predictions)
    
    print("\n🎉 分析完成！")
    print("💡 提示: 本预测仅供参考，投资有风险，请谨慎决策。")

if __name__ == "__main__":
    main()