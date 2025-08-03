#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
UNH股票未来2年走势预测分析工具
UnitedHealth Group (UNH) Stock Price Prediction for Next 2 Years
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class UNHStockPredictor:
    def __init__(self):
        self.symbol = "UNH"
        self.data = None
        self.predictions = None
        
    def fetch_data(self, period="5y"):
        """获取UNH股票历史数据"""
        print(f"正在获取 {self.symbol} 股票数据...")
        try:
            ticker = yf.Ticker(self.symbol)
            self.data = ticker.history(period=period)
            print(f"成功获取 {len(self.data)} 条历史数据")
            return True
        except Exception as e:
            print(f"获取数据失败: {e}")
            return False
    
    def calculate_technical_indicators(self):
        """计算技术指标"""
        if self.data is None:
            print("请先获取股票数据")
            return
        
        # 移动平均线
        self.data['MA5'] = self.data['Close'].rolling(window=5).mean()
        self.data['MA20'] = self.data['Close'].rolling(window=20).mean()
        self.data['MA50'] = self.data['Close'].rolling(window=50).mean()
        self.data['MA200'] = self.data['Close'].rolling(window=200).mean()
        
        # RSI指标
        delta = self.data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        self.data['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD指标
        exp1 = self.data['Close'].ewm(span=12).mean()
        exp2 = self.data['Close'].ewm(span=26).mean()
        self.data['MACD'] = exp1 - exp2
        self.data['Signal'] = self.data['MACD'].ewm(span=9).mean()
        
        # 布林带
        self.data['BB_middle'] = self.data['Close'].rolling(window=20).mean()
        bb_std = self.data['Close'].rolling(window=20).std()
        self.data['BB_upper'] = self.data['BB_middle'] + (bb_std * 2)
        self.data['BB_lower'] = self.data['BB_middle'] - (bb_std * 2)
        
        # 成交量指标
        self.data['Volume_MA'] = self.data['Volume'].rolling(window=20).mean()
        
        print("技术指标计算完成")
    
    def analyze_fundamentals(self):
        """基本面分析"""
        print("\n=== UNH基本面分析 ===")
        
        # 获取基本信息
        ticker = yf.Ticker(self.symbol)
        info = ticker.info
        
        print(f"公司名称: {info.get('longName', 'N/A')}")
        print(f"行业: {info.get('industry', 'N/A')}")
        print(f"市值: ${info.get('marketCap', 0):,.0f}")
        print(f"市盈率: {info.get('trailingPE', 'N/A')}")
        print(f"市净率: {info.get('priceToBook', 'N/A')}")
        print(f"股息收益率: {info.get('dividendYield', 0)*100:.2f}%")
        print(f"52周最高: ${info.get('fiftyTwoWeekHigh', 0):.2f}")
        print(f"52周最低: ${info.get('fiftyTwoWeekLow', 0):.2f}")
        
        # 财务指标
        print(f"\n财务指标:")
        print(f"总资产: ${info.get('totalAssets', 0):,.0f}")
        print(f"总负债: ${info.get('totalDebt', 0):,.0f}")
        print(f"净利润率: {info.get('profitMargins', 0)*100:.2f}%")
        print(f"股本回报率: {info.get('returnOnEquity', 0)*100:.2f}%")
    
    def predict_future_prices(self, days=730):  # 2年 = 730天
        """预测未来价格"""
        if self.data is None:
            print("请先获取股票数据")
            return
        
        # 使用多种方法进行预测
        predictions = {}
        
        # 方法1: 线性回归
        x = np.arange(len(self.data))
        y = self.data['Close'].values
        coeffs = np.polyfit(x, y, 1)
        future_x = np.arange(len(self.data), len(self.data) + days)
        predictions['Linear'] = np.polyval(coeffs, future_x)
        
        # 方法2: 指数平滑
        alpha = 0.1
        last_price = self.data['Close'].iloc[-1]
        exp_predictions = []
        for i in range(days):
            if i == 0:
                exp_predictions.append(last_price)
            else:
                new_price = alpha * last_price + (1 - alpha) * exp_predictions[-1]
                exp_predictions.append(new_price)
        predictions['Exponential'] = exp_predictions
        
        # 方法3: 移动平均趋势
        ma_trend = (self.data['MA50'].iloc[-1] - self.data['MA50'].iloc[-50]) / 50
        ma_predictions = []
        current_price = self.data['Close'].iloc[-1]
        for i in range(days):
            new_price = current_price + (ma_trend * (i + 1))
            ma_predictions.append(new_price)
        predictions['MA_Trend'] = ma_predictions
        
        # 方法4: 蒙特卡洛模拟
        returns = self.data['Close'].pct_change().dropna()
        mean_return = returns.mean()
        std_return = returns.std()
        
        mc_predictions = []
        current_price = self.data['Close'].iloc[-1]
        for _ in range(100):  # 100次模拟
            sim_prices = [current_price]
            for _ in range(days):
                daily_return = np.random.normal(mean_return, std_return)
                new_price = sim_prices[-1] * (1 + daily_return)
                sim_prices.append(new_price)
            mc_predictions.append(sim_prices[1:])
        
        predictions['Monte_Carlo'] = np.array(mc_predictions)
        
        self.predictions = predictions
        return predictions
    
    def generate_forecast_report(self):
        """生成预测报告"""
        if self.predictions is None:
            print("请先进行价格预测")
            return
        
        print("\n" + "="*60)
        print("UNH股票未来2年走势预测报告")
        print("="*60)
        
        # 当前价格
        current_price = self.data['Close'].iloc[-1]
        print(f"\n当前价格: ${current_price:.2f}")
        
        # 各方法预测结果
        print("\n各预测方法结果:")
        for method, pred in self.predictions.items():
            if method == 'Monte_Carlo':
                mean_pred = np.mean(pred, axis=0)
                final_price = mean_pred[-1]
                confidence_interval = np.percentile(pred[:, -1], [25, 75])
            else:
                final_price = pred[-1]
                confidence_interval = None
            
            change_pct = ((final_price - current_price) / current_price) * 100
            print(f"{method:15}: ${final_price:.2f} ({change_pct:+.2f}%)")
            
            if confidence_interval is not None:
                print(f"{'':15}  置信区间: ${confidence_interval[0]:.2f} - ${confidence_interval[1]:.2f}")
        
        # 综合预测
        methods = ['Linear', 'Exponential', 'MA_Trend']
        final_prices = [self.predictions[method][-1] for method in methods]
        avg_final_price = np.mean(final_prices)
        avg_change_pct = ((avg_final_price - current_price) / current_price) * 100
        
        print(f"\n综合预测:")
        print(f"平均目标价: ${avg_final_price:.2f}")
        print(f"预期涨幅: {avg_change_pct:+.2f}%")
        
        # 风险评估
        print(f"\n风险评估:")
        if avg_change_pct > 20:
            risk_level = "低风险 - 强烈看涨"
        elif avg_change_pct > 10:
            risk_level = "中低风险 - 看涨"
        elif avg_change_pct > 0:
            risk_level = "中等风险 - 温和看涨"
        elif avg_change_pct > -10:
            risk_level = "中高风险 - 看跌"
        else:
            risk_level = "高风险 - 强烈看跌"
        
        print(f"风险等级: {risk_level}")
        
        # 投资建议
        print(f"\n投资建议:")
        if avg_change_pct > 15:
            print("- 建议买入或持有")
            print("- 关注技术指标确认信号")
        elif avg_change_pct > 5:
            print("- 建议谨慎买入")
            print("- 分批建仓，设置止损")
        elif avg_change_pct > -5:
            print("- 建议观望")
            print("- 等待更好的入场机会")
        else:
            print("- 建议卖出或观望")
            print("- 等待价格企稳后再考虑")
    
    def plot_analysis(self):
        """绘制分析图表"""
        if self.data is None:
            print("请先获取股票数据")
            return
        
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        fig.suptitle('UNH股票技术分析与预测', fontsize=16, fontweight='bold')
        
        # 1. 价格走势图
        ax1 = axes[0, 0]
        ax1.plot(self.data.index, self.data['Close'], label='收盘价', linewidth=2)
        ax1.plot(self.data.index, self.data['MA20'], label='MA20', alpha=0.7)
        ax1.plot(self.data.index, self.data['MA50'], label='MA50', alpha=0.7)
        ax1.plot(self.data.index, self.data['MA200'], label='MA200', alpha=0.7)
        ax1.fill_between(self.data.index, self.data['BB_upper'], self.data['BB_lower'], 
                        alpha=0.3, label='布林带')
        ax1.set_title('价格走势与技术指标')
        ax1.set_ylabel('价格 ($)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. RSI指标
        ax2 = axes[0, 1]
        ax2.plot(self.data.index, self.data['RSI'], label='RSI', color='purple')
        ax2.axhline(y=70, color='r', linestyle='--', alpha=0.5, label='超买线')
        ax2.axhline(y=30, color='g', linestyle='--', alpha=0.5, label='超卖线')
        ax2.set_title('RSI指标')
        ax2.set_ylabel('RSI')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. MACD指标
        ax3 = axes[1, 0]
        ax3.plot(self.data.index, self.data['MACD'], label='MACD', color='blue')
        ax3.plot(self.data.index, self.data['Signal'], label='Signal', color='red')
        ax3.bar(self.data.index, self.data['MACD'] - self.data['Signal'], 
               alpha=0.3, label='MACD柱状图')
        ax3.set_title('MACD指标')
        ax3.set_ylabel('MACD')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. 成交量
        ax4 = axes[1, 1]
        ax4.bar(self.data.index, self.data['Volume'], alpha=0.6, label='成交量')
        ax4.plot(self.data.index, self.data['Volume_MA'], color='red', 
                label='成交量MA', linewidth=2)
        ax4.set_title('成交量分析')
        ax4.set_ylabel('成交量')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 5. 预测结果
        if self.predictions is not None:
            ax5 = axes[2, 0]
            future_dates = pd.date_range(start=self.data.index[-1], periods=731, freq='D')[1:]
            
            # 绘制历史数据
            ax5.plot(self.data.index, self.data['Close'], label='历史价格', linewidth=2)
            
            # 绘制预测线
            colors = ['red', 'blue', 'green', 'orange']
            for i, (method, pred) in enumerate(self.predictions.items()):
                if method == 'Monte_Carlo':
                    mean_pred = np.mean(pred, axis=0)
                    ax5.plot(future_dates[:len(mean_pred)], mean_pred, 
                            label=f'{method}预测', color=colors[i], linestyle='--')
                else:
                    ax5.plot(future_dates[:len(pred)], pred, 
                            label=f'{method}预测', color=colors[i], linestyle='--')
            
            ax5.set_title('未来2年价格预测')
            ax5.set_ylabel('价格 ($)')
            ax5.legend()
            ax5.grid(True, alpha=0.3)
        
        # 6. 蒙特卡洛模拟分布
        if self.predictions is not None and 'Monte_Carlo' in self.predictions:
            ax6 = axes[2, 1]
            final_prices = self.predictions['Monte_Carlo'][:, -1]
            ax6.hist(final_prices, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
            ax6.axvline(np.mean(final_prices), color='red', linestyle='--', 
                       label=f'平均: ${np.mean(final_prices):.2f}')
            ax6.set_title('蒙特卡洛模拟结果分布')
            ax6.set_xlabel('预测价格 ($)')
            ax6.set_ylabel('频次')
            ax6.legend()
            ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def run_complete_analysis(self):
        """运行完整分析"""
        print("开始UNH股票分析...")
        
        # 1. 获取数据
        if not self.fetch_data():
            return
        
        # 2. 计算技术指标
        self.calculate_technical_indicators()
        
        # 3. 基本面分析
        self.analyze_fundamentals()
        
        # 4. 预测未来价格
        self.predict_future_prices()
        
        # 5. 生成报告
        self.generate_forecast_report()
        
        # 6. 绘制图表
        self.plot_analysis()
        
        print("\n分析完成！")

def main():
    """主函数"""
    print("UNH股票未来2年走势预测分析工具")
    print("="*50)
    
    predictor = UNHStockPredictor()
    predictor.run_complete_analysis()

if __name__ == "__main__":
    main()