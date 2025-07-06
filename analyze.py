import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def create_modern_dxy_2_0():
    """
    Create DXY 2.0 - A modern, trade-weighted dollar strength index
    """
    
    print("🚀 Creating DXY 2.0: Modern Trade-Weighted Dollar Index")
    print("=" * 65)
    
    # Modern trade-weighted currency basket (exactly 100%)
    weights = {
        'EUR': 17.0,  # Euro - EU trade
        'MXN': 16.0,  # Mexican Peso - largest individual partner
        'CAD': 15.0,  # Canadian Dollar - 2nd largest partner
        'CNY': 12.0,  # Chinese Yuan - 3rd largest partner
        'JPY': 5.0,   # Japanese Yen - traditional major partner
        'GBP': 4.5,   # British Pound - financial partner
        'KRW': 4.0,   # Korean Won - major Asian partner
        'TWD': 3.5,   # Taiwan Dollar - tech trade partner
        'INR': 3.0,   # Indian Rupee - growing relationship
        'BRL': 3.0,   # Brazilian Real - commodity trade
        'AUD': 3.0,   # Australian Dollar - commodity trade
        'CHF': 2.5,   # Swiss Franc - financial center
        'SGD': 2.5,   # Singapore Dollar - financial center
        'SEK': 2.0,   # Swedish Krona - Nordic representation
        'VND': 2.0,   # Vietnamese Dong - Asian trade
        'NOK': 2.0,   # Norwegian Krone - Nordic trade
        'NZD': 1.5,   # New Zealand Dollar - Pacific trade
        'THB': 1.5,   # Thai Baht - Southeast Asia trade
    }
    
    print("📊 Currency Basket Composition:")
    for currency, weight in sorted(weights.items(), key=lambda x: x[1], reverse=True):
        print(f"  {currency}: {weight:5.1f}%")
    print(f"  Total: {sum(weights.values()):5.1f}%")
    
    # Load currency data
    print(f"\n💱 Loading 12-month currency data...")
    all_data = {}
    
    for currency in weights.keys():
        filename = f'currency_data/{currency}_USD_12m.csv'
        if os.path.exists(filename):
            df = pd.read_csv(filename)
            df['Date'] = pd.to_datetime(df['Date'], utc=True)
            df = df.set_index('Date')
            all_data[currency] = df['Close'].copy()
            print(f"  ✅ {currency}: {len(df)} rows")
        else:
            print(f"  ❌ Missing {currency}")
            return None
    
    # Create combined dataset
    combined_df = pd.DataFrame(all_data)
    combined_df = combined_df.dropna()
    
    print(f"\n📈 Dataset Summary:")
    print(f"  Rows: {len(combined_df)}")
    print(f"  Date range: {combined_df.index.min().strftime('%Y-%m-%d')} to {combined_df.index.max().strftime('%Y-%m-%d')}")
    print(f"  Trading days: {len(combined_df)}")
    
    # Calculate weighted basket rate
    weighted_rates = pd.Series(0.0, index=combined_df.index)
    
    print(f"\n💰 Individual Currency Performance:")
    currency_performance = {}
    
    for currency, weight in weights.items():
        rates = combined_df[currency]
        weighted_rates += rates * (weight / 100)
        
        start_rate = rates.iloc[0]
        end_rate = rates.iloc[-1]
        change_pct = ((end_rate - start_rate) / start_rate) * 100
        currency_performance[currency] = change_pct
        
        direction = "🔴" if change_pct > 0 else "🟢"  # Red = currency up = dollar down
        print(f"  {currency}: {change_pct:+7.2f}% (weight: {weight:4.1f}%) {direction}")
    
    # Create DXY 2.0 index (higher = stronger dollar)
    base_rate = weighted_rates.iloc[0]
    dxy_2_0 = 100 * (base_rate / weighted_rates)
    
    # Performance metrics
    start_dxy = dxy_2_0.iloc[0]
    end_dxy = dxy_2_0.iloc[-1]
    total_change = ((end_dxy - start_dxy) / start_dxy) * 100
    max_dxy = dxy_2_0.max()
    min_dxy = dxy_2_0.min()
    volatility = dxy_2_0.std()
    
    print(f"\n🎯 DXY 2.0 Performance Summary:")
    print(f"  Starting Value: {start_dxy:.2f}")
    print(f"  Ending Value:   {end_dxy:.2f}")
    print(f"  Total Change:   {total_change:+.2f}%")
    print(f"  Peak Value:     {max_dxy:.2f}")
    print(f"  Trough Value:   {min_dxy:.2f}")
    print(f"  Volatility:     {volatility:.2f}")
    
    # Dollar strength interpretation
    if total_change < -2:
        strength = "SIGNIFICANTLY WEAKENED"
        emoji = "📉🔴"
    elif total_change < 0:
        strength = "WEAKENED"
        emoji = "📉"
    elif total_change > 2:
        strength = "SIGNIFICANTLY STRENGTHENED" 
        emoji = "📈🟢"
    else:
        strength = "STRENGTHENED"
        emoji = "📈"
    
    print(f"\n{emoji} VERDICT: The US Dollar has {strength} by {abs(total_change):.2f}% over the past 12 months")
    print(f"    against this modern trade-weighted currency basket.")
    
    # Identify top contributors to dollar weakness/strength
    print(f"\n🔍 Key Contributors:")
    sorted_performance = sorted(currency_performance.items(), key=lambda x: x[1], reverse=True)
    
    print("  Most Dollar-Negative (currency gained vs USD):")
    for currency, perf in sorted_performance[:3]:
        weight = weights[currency]
        contribution = perf * (weight / 100)
        print(f"    {currency}: {perf:+.2f}% × {weight:.1f}% weight = {contribution:+.3f}% impact")
    
    print("  Most Dollar-Positive (currency lost vs USD):")
    for currency, perf in sorted_performance[-3:]:
        weight = weights[currency]
        contribution = perf * (weight / 100)
        print(f"    {currency}: {perf:+.2f}% × {weight:.1f}% weight = {contribution:+.3f}% impact")
    
    # Create results dataframe
    results_df = pd.DataFrame({
        'Date': combined_df.index,
        'DXY_2_0': dxy_2_0,
        'Weighted_Rate': weighted_rates
    })
    
    return results_df, weights, currency_performance

def create_analysis_visualization(results_df, weights, currency_performance):
    """
    Create comprehensive DXY 2.0 analysis charts
    """
    
    print(f"\n📊 Creating DXY 2.0 analysis visualization...")
    
    # Set up the figure
    fig = plt.figure(figsize=(20, 14))
    gs = fig.add_gridspec(3, 3, height_ratios=[2, 1, 1], width_ratios=[2, 1, 1])
    
    # Main DXY 2.0 chart (large, top)
    ax_main = fig.add_subplot(gs[0, :])
    
    # Plot DXY 2.0 with fill
    dates = pd.to_datetime(results_df['Date'])
    values = results_df['DXY_2_0']
    
    ax_main.plot(dates, values, linewidth=3, color='#1f77b4', label='DXY 2.0')
    ax_main.axhline(y=100, color='red', linestyle='--', alpha=0.8, linewidth=2, label='Baseline (100)')
    
    # Fill areas
    ax_main.fill_between(dates, values, 100, 
                        where=(values > 100), color='green', alpha=0.3, label='Dollar Strength')
    ax_main.fill_between(dates, values, 100, 
                        where=(values <= 100), color='red', alpha=0.3, label='Dollar Weakness')
    
    # Add key statistics
    start_val = values.iloc[0]
    end_val = values.iloc[-1]
    change_pct = ((end_val - start_val) / start_val) * 100
    max_val = values.max()
    min_val = values.min()
    
    # Title and labels
    ax_main.set_title('DXY 2.0: Modern Trade-Weighted US Dollar Strength Index', 
                     fontsize=18, fontweight='bold', pad=20)
    ax_main.set_ylabel('Index Value (Base = 100)', fontsize=14)
    ax_main.legend(loc='upper right', fontsize=12)
    ax_main.grid(True, alpha=0.3)
    
    # Statistics box
    stats_text = f"""12-Month Performance:
Change: {change_pct:+.2f}%
Peak: {max_val:.1f}
Trough: {min_val:.1f}
Current: {end_val:.1f}"""
    
    ax_main.text(0.02, 0.98, stats_text, transform=ax_main.transAxes, 
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8),
                fontsize=12, fontweight='bold', verticalalignment='top')
    
    # Currency weights pie chart
    ax_pie = fig.add_subplot(gs[1, 0])
    colors = plt.cm.Set3(np.linspace(0, 1, len(weights)))
    wedges, texts, autotexts = ax_pie.pie(weights.values(), labels=weights.keys(), 
                                         autopct='%1.1f%%', colors=colors, startangle=90)
    ax_pie.set_title('Currency Weights', fontsize=14, fontweight='bold')
    
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(10)
    
    # Currency performance bar chart
    ax_bar = fig.add_subplot(gs[1, 1:])
    currencies = list(currency_performance.keys())
    performances = list(currency_performance.values())
    
    colors_bar = ['red' if p > 0 else 'green' for p in performances]
    bars = ax_bar.barh(currencies, performances, color=colors_bar, alpha=0.7)
    ax_bar.axvline(x=0, color='black', linestyle='-', alpha=0.8)
    ax_bar.set_xlabel('% Change vs USD (12 months)', fontsize=12)
    ax_bar.set_title('Individual Currency Performance', fontsize=14, fontweight='bold')
    ax_bar.grid(True, alpha=0.3, axis='x')
    
    # Add value labels on bars
    for bar, perf in zip(bars, performances):
        width = bar.get_width()
        ax_bar.text(width + (0.3 if width > 0 else -0.3), bar.get_y() + bar.get_height()/2, 
                   f'{perf:+.1f}%', ha='left' if width > 0 else 'right', va='center', fontsize=10)
    
    # DXY 2.0 trend analysis
    ax_trend = fig.add_subplot(gs[2, :])
    
    # Calculate rolling 30-day change
    rolling_change = values.rolling(window=30).apply(lambda x: ((x.iloc[-1] - x.iloc[0]) / x.iloc[0]) * 100)
    
    ax_trend.plot(dates, rolling_change, linewidth=2, color='purple', alpha=0.8)
    ax_trend.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    ax_trend.fill_between(dates, rolling_change, 0, 
                         where=(rolling_change > 0), color='green', alpha=0.3)
    ax_trend.fill_between(dates, rolling_change, 0, 
                         where=(rolling_change <= 0), color='red', alpha=0.3)
    
    ax_trend.set_title('30-Day Rolling Dollar Strength Change (%)', fontsize=14, fontweight='bold')
    ax_trend.set_ylabel('% Change', fontsize=12)
    ax_trend.set_xlabel('Date', fontsize=12)
    ax_trend.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('output/DXY_2_0_Complete_Analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✅ Complete analysis saved as 'DXY_2_0_Complete_Analysis.png'")

def save_results(results_df, weights, currency_performance):
    """
    Save all results to CSV files
    """
    
    print(f"\n💾 Saving results...")
    
    # Main index data
    results_df.to_csv('output/DXY_2_0_Final.csv', index=False)
    
    # Currency weights
    weights_df = pd.DataFrame.from_dict(weights, orient='index', columns=['Weight_%'])
    weights_df.index.name = 'Currency'
    weights_df.to_csv('output/DXY_2_0_Weights_Final.csv')
    
    # Currency performance summary
    performance_df = pd.DataFrame.from_dict(currency_performance, orient='index', columns=['Change_%'])
    performance_df['Weight_%'] = [weights[curr] for curr in performance_df.index]
    performance_df['Contribution'] = performance_df['Change_%'] * performance_df['Weight_%'] / 100
    performance_df = performance_df.sort_values('Change_%', ascending=False)
    performance_df.index.name = 'Currency'
    performance_df.to_csv('output/DXY_2_0_Currency_Analysis.csv')
    
    print("✅ Files saved:")
    print("  - DXY_2_0_Final.csv (main index data)")
    print("  - DXY_2_0_Weights_Final.csv (currency weights)")
    print("  - DXY_2_0_Currency_Analysis.csv (individual currency analysis)")

if __name__ == "__main__":
    print("🚀 DXY 2.0: Modern Dollar Strength Analysis")
    print("💡 No comparison to legacy DXY - this IS the better metric!\n")
    
    # Create the modern index
    results_df, weights, currency_performance = create_modern_dxy_2_0()
    
    if results_df is not None:
        # Save all results
        save_results(results_df, weights, currency_performance)

        # Create comprehensive visualization
        create_analysis_visualization(results_df, weights, currency_performance)
        
        print(f"\n" + "🎯" * 20)
        print("DXY 2.0 ANALYSIS COMPLETE!")
        print("This modern trade-weighted index better reflects")
        print("actual US dollar strength vs current trading partners.")
        print("🎯" * 20)
    else:
        print("❌ Failed to create DXY 2.0 analysis")
