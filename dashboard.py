import dash
from dash import html, dcc, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sentiment_analyzer import SentimentAnalyzer
from emotion_classifier import EmotionClassifier
from contradiction_detector import ContradictionDetector
from data_generator import DataGenerator

# Initialize the Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Initialize analyzers
sentiment_analyzer = SentimentAnalyzer()
emotion_classifier = EmotionClassifier()
contradiction_detector = ContradictionDetector()
data_generator = DataGenerator()

# Generate initial synthetic data
df = data_generator.generate_synthetic_data(100)
data_generator.save_to_csv(df)

# Define the layout
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H1("Greenwashing Detection Dashboard", className="text-center mb-4"),
            html.Hr()
        ])
    ]),
    
    # Data Upload Section
    dbc.Row([
        dbc.Col([
            html.H4("Data Upload", className="mb-3"),
            dcc.Upload(
                id='upload-data',
                children=dbc.Button('Upload CSV File', color='primary'),
                multiple=False
            ),
            html.Div(id='upload-status', className="mt-2")
        ])
    ], className="mb-4"),
    
    # Filters Section
    dbc.Row([
        dbc.Col([
            html.H4("Filters", className="mb-3"),
            dbc.Row([
                dbc.Col([
                    html.Label("Company"),
                    dcc.Dropdown(
                        id='company-filter',
                        options=[{'label': company, 'value': company} for company in df['company'].unique()],
                        value=df['company'].unique()[0]
                    )
                ], width=6),
                dbc.Col([
                    html.Label("Platform"),
                    dcc.Dropdown(
                        id='platform-filter',
                        options=[{'label': platform, 'value': platform} for platform in df['platform'].unique()],
                        value=df['platform'].unique()[0]
                    )
                ], width=6)
            ]),
            dbc.Row([
                dbc.Col([
                    html.Label("Date Range"),
                    dcc.DatePickerRange(
                        id='date-range',
                        start_date=df['timestamp'].min(),
                        end_date=df['timestamp'].max()
                    )
                ], width=12)
            ], className="mt-3")
        ])
    ], className="mb-4"),
    
    # Main Analysis Section
    dbc.Row([
        # Sentiment Analysis
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Sentiment Analysis"),
                dbc.CardBody([
                    dcc.Graph(id='sentiment-trend'),
                    dcc.Graph(id='sentiment-distribution')
                ])
            ])
        ], width=6),
        
        # Emotion Classification
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Emotion Classification"),
                dbc.CardBody([
                    dcc.Graph(id='emotion-distribution'),
                    dcc.Graph(id='emotion-trend')
                ])
            ])
        ], width=6)
    ], className="mb-4"),
    
    # Contradiction Analysis
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Contradiction Analysis"),
                dbc.CardBody([
                    dcc.Graph(id='contradiction-score'),
                    dcc.Graph(id='greenwashing-risk')
                ])
            ])
        ], width=12)
    ], className="mb-4"),
    
    # Detailed Analysis Section
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Detailed Analysis"),
                dbc.CardBody([
                    html.Div(id='detailed-analysis')
                ])
            ])
        ], width=12)
    ])
], fluid=True)

# Callback for data upload
@app.callback(
    [Output('upload-status', 'children'),
     Output('company-filter', 'options'),
     Output('platform-filter', 'options')],
    Input('upload-data', 'contents'),
    State('upload-data', 'filename')
)
def update_output(contents, filename):
    if contents is None:
        return "No file uploaded", [], []
    
    try:
        df = pd.read_csv(filename)
        return (
            f"Successfully uploaded {filename}",
            [{'label': company, 'value': company} for company in df['company'].unique()],
            [{'label': platform, 'value': platform} for platform in df['platform'].unique()]
        )
    except Exception as e:
        return f"Error uploading file: {str(e)}", [], []

# Callback for sentiment trend
@app.callback(
    Output('sentiment-trend', 'figure'),
    [Input('company-filter', 'value'),
     Input('platform-filter', 'value'),
     Input('date-range', 'start_date'),
     Input('date-range', 'end_date')]
)
def update_sentiment_trend(company, platform, start_date, end_date):
    filtered_df = df[
        (df['company'] == company) &
        (df['platform'] == platform) &
        (df['timestamp'] >= start_date) &
        (df['timestamp'] <= end_date)
    ]
    
    fig = px.line(
        filtered_df,
        x='timestamp',
        y='sentiment_score',
        title='Sentiment Score Trend'
    )
    return fig

# Callback for sentiment distribution
@app.callback(
    Output('sentiment-distribution', 'figure'),
    [Input('company-filter', 'value'),
     Input('platform-filter', 'value')]
)
def update_sentiment_distribution(company, platform):
    filtered_df = df[
        (df['company'] == company) &
        (df['platform'] == platform)
    ]
    
    fig = px.histogram(
        filtered_df,
        x='sentiment_score',
        title='Sentiment Score Distribution'
    )
    return fig

# Callback for emotion distribution
@app.callback(
    Output('emotion-distribution', 'figure'),
    [Input('company-filter', 'value'),
     Input('platform-filter', 'value')]
)
def update_emotion_distribution(company, platform):
    filtered_df = df[
        (df['company'] == company) &
        (df['platform'] == platform)
    ]
    
    # Analyze emotions for each post
    emotions = []
    for content in filtered_df['content']:
        result = emotion_classifier.get_sustainability_emotions(content)
        if result['has_sustainability_context']:
            emotions.append(result['primary_emotion'])
    
    emotion_counts = pd.Series(emotions).value_counts()
    
    fig = px.pie(
        values=emotion_counts.values,
        names=emotion_counts.index,
        title='Emotion Distribution'
    )
    return fig

# Callback for contradiction score
@app.callback(
    Output('contradiction-score', 'figure'),
    [Input('company-filter', 'value'),
     Input('platform-filter', 'value')]
)
def update_contradiction_score(company, platform):
    filtered_df = df[
        (df['company'] == company) &
        (df['platform'] == platform)
    ]
    
    fig = px.box(
        filtered_df,
        y='contradiction_score',
        title='Contradiction Score Distribution'
    )
    return fig

# Callback for greenwashing risk
@app.callback(
    Output('greenwashing-risk', 'figure'),
    [Input('company-filter', 'value'),
     Input('platform-filter', 'value')]
)
def update_greenwashing_risk(company, platform):
    filtered_df = df[
        (df['company'] == company) &
        (df['platform'] == platform)
    ]
    
    # Check if we have any data
    if len(filtered_df) == 0:
        # Return empty figure with message
        fig = go.Figure()
        fig.add_annotation(
            text="No data available for selected filters",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False
        )
        fig.update_layout(
            title='Greenwashing Risk Analysis',
            xaxis_title='Sentiment Score',
            yaxis_title='Contradiction Score'
        )
        return fig
    
    # Calculate greenwashing risk score
    risk_scores = []
    for _, row in filtered_df.iterrows():
        sentiment = row['sentiment_score']
        contradiction = row['contradiction_score']
        risk_score = (abs(sentiment) + contradiction) / 2
        risk_scores.append(risk_score)
    
    # Convert risk scores to numpy array and normalize for better visualization
    risk_scores = np.array(risk_scores)
    normalized_sizes = (risk_scores - risk_scores.min()) / (risk_scores.max() - risk_scores.min()) * 20 + 5  # Scale between 5 and 25
    
    fig = px.scatter(
        filtered_df,
        x='sentiment_score',
        y='contradiction_score',
        title='Greenwashing Risk Analysis'
    )
    
    # Update marker sizes
    fig.update_traces(
        marker=dict(
            size=normalized_sizes,
            color=risk_scores,
            colorscale='RdYlGn_r',  # Red to Green color scale (red for high risk)
            showscale=True,
            colorbar=dict(title='Risk Score')
        )
    )
    
    # Update axis labels
    fig.update_layout(
        xaxis_title='Sentiment Score',
        yaxis_title='Contradiction Score'
    )
    
    return fig

# Callback for detailed analysis
@app.callback(
    Output('detailed-analysis', 'children'),
    [Input('company-filter', 'value'),
     Input('platform-filter', 'value')]
)
def update_detailed_analysis(company, platform):
    filtered_df = df[
        (df['company'] == company) &
        (df['platform'] == platform)
    ]
    
    # Calculate key metrics
    avg_sentiment = filtered_df['sentiment_score'].mean()
    avg_contradiction = filtered_df['contradiction_score'].mean()
    total_posts = len(filtered_df)
    
    # Analyze emotions
    emotions = []
    for content in filtered_df['content']:
        result = emotion_classifier.get_sustainability_emotions(content)
        if result['has_sustainability_context']:
            emotions.extend(result['emotions'].keys())
    
    top_emotions = pd.Series(emotions).value_counts().head(3)
    
    return html.Div([
        html.H5("Key Metrics"),
        html.Ul([
            html.Li(f"Average Sentiment Score: {avg_sentiment:.2f}"),
            html.Li(f"Average Contradiction Score: {avg_contradiction:.2f}"),
            html.Li(f"Total Posts Analyzed: {total_posts}")
        ]),
        html.H5("Top Emotions"),
        html.Ul([
            html.Li(f"{emotion}: {count} occurrences")
            for emotion, count in top_emotions.items()
        ])
    ])

if __name__ == '__main__':
    app.run(debug=True) 