import plotly.graph_objects as go
import numpy as np

def generate_live_chart(data, monday_high, monday_low, buy_signals, sell_signals):
    """
    Generate an interactive Plotly candlestick chart with:
      - Monday high/low horizontal lines
      - A trend line (using linear regression on the close prices)
      - Buy/sell signal markers
      - A TradingView-like dark theme with zoom/pan enabled
    """
    try:
        if data is None or data.empty:
            print("No data available for chart generation")
            fig = go.Figure()
            fig.add_annotation(text="No data available", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
            return fig

        # Create the basic candlestick chart
        fig = go.Figure(data=[go.Candlestick(
            x=data.index,
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Close'],
            name="BTC-USD"
        )])

        # Add horizontal lines for Monday levels if available
        if monday_high is not None:
            fig.add_hline(
                y=monday_high,
                line=dict(color="red", dash="dash"),
                annotation_text="Monday High",
                annotation_position="top left"
            )
        if monday_low is not None:
            fig.add_hline(
                y=monday_low,
                line=dict(color="green", dash="dash"),
                annotation_text="Monday Low",
                annotation_position="bottom left"
            )

        # Calculate and add trend line if enough data points
        if len(data) > 1:
            try:
                x_numeric = np.arange(len(data.index))
                y = data['Close'].values
                slope, intercept = np.polyfit(x_numeric, y, 1)
                trend_line = slope * x_numeric + intercept
                fig.add_trace(go.Scatter(
                    x=data.index,
                    y=trend_line,
                    mode="lines",
                    name="Trend Line",
                    line=dict(color="orange", width=2)
                ))
            except Exception as e:
                print(f"Error calculating trend line: {e}")

        # Add buy signal markers if available
        if buy_signals and len(buy_signals) > 0:
            try:
                buy_times = [signal[0] for signal in buy_signals]
                buy_prices = [signal[1] for signal in buy_signals]
                fig.add_trace(go.Scatter(
                    x=buy_times,
                    y=buy_prices,
                    mode="markers",
                    marker=dict(symbol="triangle-up", size=12, color="lime"),
                    name="Buy Signal"
                ))
            except Exception as e:
                print(f"Error adding buy signals: {e}")

        # Add sell signal markers if available
        if sell_signals and len(sell_signals) > 0:
            try:
                sell_times = [signal[0] for signal in sell_signals]
                sell_prices = [signal[1] for signal in sell_signals]
                fig.add_trace(go.Scatter(
                    x=sell_times,
                    y=sell_prices,
                    mode="markers",
                    marker=dict(symbol="triangle-down", size=12, color="yellow"),
                    name="Sell Signal"
                ))
            except Exception as e:
                print(f"Error adding sell signals: {e}")

        # Update layout to mimic TradingView style
        fig.update_layout(
            title="BTC-USD Live Trading Chart",
            xaxis_title="Time",
            yaxis_title="Price (USD)",
            template="plotly_dark",
            xaxis=dict(
                rangeslider=dict(visible=False),
                type="date",
                fixedrange=False
            ),
            yaxis=dict(
                fixedrange=False
            ),
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            )
        )

        return fig

    except Exception as e:
        print(f"Error generating chart: {e}")
        fig = go.Figure()
        fig.add_annotation(text=f"Error generating chart", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return fig