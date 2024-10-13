import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import os
import json

app = dash.Dash(__name__)

# Define available names based on training_results folder
available_names = [f for f in os.listdir('training_results') if os.path.isdir(os.path.join('training_results', f))]
name_generations = {
    name: sorted(
        [gen for gen in os.listdir(os.path.join('training_results', name)) if os.path.isdir(os.path.join('training_results', name, gen))],
        key=lambda x: int(x.split('_')[-1])
    )
    for name in available_names
}

app.layout = html.Div(
    style={'width': '100vw', 'height': '100vh', 'display': 'flex', 'flexDirection': 'column'},
    children=[
        html.H3("NEAT Model Trading Performance", style={'text-align': 'center'}),

        # Wrapper Div for Dropdown and Generation Output in the same row
        html.Div(
            style={'display': 'flex', 'flexDirection': 'row', 'alignItems': 'center', 'padding': '10px'},
            children=[
                dcc.Dropdown(
                    id='name-select', 
                    options=[{'label': name, 'value': name} for name in available_names], 
                    value=available_names[0],
                    style={'flex': '1', 'marginRight': '10px'}
                ),
                html.Div(id='generation-output', style={'fontSize': 20, 'text-align': 'center', 'padding': '10px', 'flex': '0.5'})
            ]
        ),

        html.Div(
            dcc.Slider(
                id='generation-slider', 
                min=0, 
                max=0, 
                step=1, 
                value=0, 
                marks={i: str(i) for i in range(0, 11)}  # Adjust marks as needed
            ),
            style={'padding': '10px'}
        ),
        
        dcc.Graph(
            id='main-figure',
            style={'flexGrow': '1'}  # Allow the graph to expand and take up available space
        ),

        # Add Interval component for periodic updates
        dcc.Interval(
            id='interval-component',
            interval=10*60*1000,  # 10 minutes in milliseconds
            n_intervals=0  # Start at zero
        ),
    ]
)

# Unified callback to update slider max, figure, and generation text
@app.callback(
    [Output('generation-slider', 'max'), Output('main-figure', 'figure'), Output('generation-output', 'children')],
    [Input('name-select', 'value'), Input('generation-slider', 'value'), Input('interval-component', 'n_intervals')]
)
def update_content(name, generation, n_intervals):
    generations = name_generations.get(name, [])
    max_generation = len(generations) - 1

    # Set folder paths
    gen_folder = os.path.join('training_results', name, f"_Gen_{generation}")
    price_data_path = os.path.join('price_data', f"{name}.csv")
    
    # Load the metrics and trade logs for both train and test sets
    with open(os.path.join(gen_folder, 'metrics.json')) as f:
        metrics = json.load(f)

    # Load train and test trade logs
    try:
        train_trade_log_df = pd.read_csv(os.path.join(gen_folder, 'train_trade_log.csv'))
        test_trade_log_df = pd.read_csv(os.path.join(gen_folder, 'test_trade_log.csv'))
    except (pd.errors.EmptyDataError, FileNotFoundError):
        train_trade_log_df = pd.DataFrame()
        test_trade_log_df = pd.DataFrame()

    price_data_df = pd.read_csv(price_data_path)
    
    # Convert Date_Time columns to datetime format
    price_data_df['Date_Time'] = pd.to_datetime(price_data_df['Date_Time'])
    
    # If both trade logs are empty, show a no trades message
    if train_trade_log_df.empty and test_trade_log_df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No trades to show",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=20)
        )
        generation_text = f"Gen {generation}"
        return max_generation, fig, generation_text
    
    # Filter price data based on trade log date ranges
    trade_logs = [train_trade_log_df, test_trade_log_df]
    filtered_price_data = []
    
    for trade_log in trade_logs:
        if not trade_log.empty:
            trade_log['Date_Time'] = pd.to_datetime(trade_log['Date_Time'])
            min_date = trade_log['Date_Time'].min()
            max_date = trade_log['Date_Time'].max()
            filtered_price_data.append(price_data_df[(price_data_df['Date_Time'] >= min_date) & (price_data_df['Date_Time'] <= max_date)])
        else:
            filtered_price_data.append(price_data_df)  # If empty, use full price data as a placeholder
    
    # Create the main figure
    fig = create_figure(filtered_price_data[0], filtered_price_data[1])

    # Add price charts and agent money charts for both train and test data
    add_price_chart(fig, filtered_price_data[0], train_trade_log_df, row=2, col=1)
    add_price_chart(fig, filtered_price_data[1], test_trade_log_df, row=2, col=2)
    
    add_agent_money_chart(fig, train_trade_log_df, row=3, col=1)
    add_agent_money_chart(fig, test_trade_log_df, row=3, col=2)
    
    # Add metrics tables
    add_metrics_table(fig, [metrics['train'][k] for k in metrics['train']], col=1, row=1)
    add_metrics_table(fig, [metrics['test'][k] for k in metrics['test']], col=2, row=1)

    # Add trade history tables
    add_trade_history_table(fig, train_trade_log_df, row=4, col=1)
    add_trade_history_table(fig, test_trade_log_df, row=4, col=2)
    
    fig.update_layout(title_text=f'Trade History, Agent Money Over Time, and Trade Log (Generation #{generation})')
    
    generation_text = f"Gen {generation}"
    
    return max_generation, fig, generation_text

# Visualization functions, adapted for Dash app use
def add_price_chart(fig, df, trade_log_df, row, col):
    fig.add_trace(go.Scatter(x=df['Date_Time'], y=df['Close'], mode='lines', name='Close Price'), row=row, col=col)
    buy_signals = trade_log_df[trade_log_df['action'] == 'Buy']
    sell_signals = trade_log_df[trade_log_df['action'] == 'Sell']
    fig.add_trace(go.Scatter(x=buy_signals['Date_Time'], y=buy_signals['price'], mode='markers', name='Buy',
                             marker=dict(color='green', size=10, symbol='triangle-up')), row=row, col=col)
    fig.add_trace(go.Scatter(x=sell_signals['Date_Time'], y=sell_signals['price'], mode='markers', name='Sell',
                             marker=dict(color='red', size=10, symbol='triangle-down')), row=row, col=col)

def add_metrics_table(fig, metrics, col, row=1):
    headers = ['Win Loss Ratio', 'Consistency Bonus', 'Sum Percentage Gains', 'Agent PnL', 'Market PnL', 'Number of Trades', 'Trade to Candle Ratio']
    fig.add_trace(
        go.Table(
            header=dict(values=headers, fill_color='grey', align='left', font=dict(color='white', size=12)),
            cells=dict(values=metrics, align='left')
        ),
        row=row, col=col
    )

def add_agent_money_chart(fig, trade_log_df, row, col):
    fig.add_trace(
        go.Scatter(
            x=trade_log_df['Date_Time'],
            y=trade_log_df['Portfolio'],
            mode='lines',
            name='Agent Money',
            showlegend=False
        ),
        row=row, col=col
    )

def add_trade_history_table(fig, trade_log_df, row, col):
    valid_trades = trade_log_df[trade_log_df['action'] != 'N/A']
    table_data = valid_trades[['Date_Time', 'action', 'price', 'Portfolio']]
    table_data['Date_Time'] = table_data['Date_Time'].astype(str).str[:-9]
    colors = [['#ABEBC6' if action == 'Buy' else '#F5B7B1' for action in table_data['action']]]
    
    fig.add_trace(
        go.Table(
            header=dict(
                values=['Date_Time', 'Action', 'Price', 'Portfolio'],
                fill_color='grey',
                align='left',
                font=dict(color='white', size=12)
            ),
            cells=dict(
                values=[table_data.Date_Time, table_data.action, table_data.price, table_data.Portfolio],
                fill_color=colors,
                align='left'
            )
        ),
        row=row, col=col
    )

def create_figure(df_train, df_test):
    return make_subplots(
        rows=4, cols=2, shared_xaxes=True, vertical_spacing=0.04, horizontal_spacing=0.05,
        row_heights=[0.1, 0.30, 0.30, 0.30],
        subplot_titles=(
            f'Agent on TRAIN data | {str(df_train["Date_Time"].iloc[0])[:-9]} to {str(df_train["Date_Time"].iloc[-1])[:-9]}',
            f'Agent on TEST data | {str(df_test["Date_Time"].iloc[0])[:-9]} to {str(df_test["Date_Time"].iloc[-1])[:-9]}'
        ),
        specs=[
            [{"type": "table"}, {"type": "table"}],
            [{"type": "scatter"}, {"type": "scatter"}],
            [{"type": "scatter"}, {"type": "scatter"}],
            [{"type": "table"}, {"type": "table"}]
        ]
    )

if __name__ == '__main__':
    app.run_server(debug=True)
