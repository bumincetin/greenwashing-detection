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
        # Left Column - Post Analysis
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Post Analysis"),
                dbc.CardBody([
                    dcc.Graph(id='sentiment-trend'),
                    dcc.Graph(id='emotion-distribution'),
                    dcc.Graph(id='greenwashing-risk')
                ])
            ])
        ], width=8),
        
        # Right Column - Comment Analysis
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Comment Analysis"),
                dbc.CardBody([
                    dcc.Graph(id='comment-sentiment-distribution'),
                    dcc.Graph(id='comment-engagement-trend'),
                    html.Div(id='comment-summary')
                ])
            ])
        ], width=4)
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
        ])
    ]),
    
    # Add Methodology Section
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader([
                    html.H4("Methodology and Interpretation Guide", className="mb-0"),
                    dbc.Button(
                        "Show/Hide",
                        id="methodology-collapse-button",
                        className="float-right",
                        color="primary",
                        size="sm",
                    ),
                ]),
                dbc.Collapse(
                    dbc.CardBody([
                        # Sentiment Trend Analysis
                        html.Div([
                            html.H5("1. Sentiment Trend Analysis", className="text-primary"),
                            html.P([
                                html.Strong("Purpose: "),
                                "Track changes in sentiment over time to identify patterns in corporate communication."
                            ]),
                            html.H6("Methodology:", className="mt-2"),
                            html.Ul([
                                html.Li("Ensemble sentiment analysis combining multiple models:"),
                                html.Ul([
                                    html.Li("VADER (30%): Specialized for social media"),
                                    html.Li("TextBlob (20%): Pattern-based analysis"),
                                    html.Li("RoBERTa (30%): Deep learning model"),
                                    html.Li("DistilBERT (20%): Efficient transformer model")
                                ]),
                                html.Li([
                                    "Rolling average calculation: ",
                                    html.Code("window=3, min_periods=1, center=True")
                                ]),
                                html.Li("Score normalization to [-1, 1] range")
                            ]),
                            html.H6("Interpretation:", className="mt-2"),
                            html.Ul([
                                html.Li("Positive scores (> 0): Optimistic or positive statements"),
                                html.Li("Negative scores (< 0): Critical or negative statements"),
                                html.Li("Trend line shows overall sentiment direction"),
                                html.Li("Sharp changes may indicate significant policy or communication shifts")
                            ])
                        ], className="mb-4"),
                        
                        # Emotion Distribution
                        html.Div([
                            html.H5("2. Emotion Distribution", className="text-primary"),
                            html.P([
                                html.Strong("Purpose: "),
                                "Analyze emotional content in sustainability communications."
                            ]),
                            html.H6("Methodology:", className="mt-2"),
                            html.Ul([
                                html.Li("Uses GoEmotions model fine-tuned for sustainability context"),
                                html.Li("Processes each post through emotion classifier"),
                                html.Li("Identifies sustainability-related content using keyword matching"),
                                html.Li("Calculates emotion frequencies and potential greenwashing indicators")
                            ]),
                            html.H6("Interpretation:", className="mt-2"),
                            html.Ul([
                                html.Li("High optimism (> 0.8): Potential greenwashing indicator"),
                                html.Li("Excessive joy (> 0.9): May indicate lack of authenticity"),
                                html.Li("High pride (> 0.8): Possible overconfidence"),
                                html.Li("Balanced distribution suggests more authentic communication")
                            ])
                        ], className="mb-4"),
                        
                        # Greenwashing Risk Analysis
                        html.Div([
                            html.H5("3. Greenwashing Risk Analysis", className="text-primary"),
                            html.P([
                                html.Strong("Purpose: "),
                                "Identify potential greenwashing through multi-factor analysis."
                            ]),
                            html.H6("Methodology:", className="mt-2"),
                            html.Ul([
                                html.Li([
                                    "Risk Score Calculation:",
                                    html.Code("risk_score = (|sentiment| + contradiction + comment_risk) / 3")
                                ]),
                                html.Li([
                                    "Comment Risk:",
                                    html.Code("(negative + 0.5 * skeptical) / total_comments")
                                ]),
                                html.Li("Point size normalized between 5-25 based on risk score"),
                                html.Li("Color scale: Green (low risk) to Red (high risk)")
                            ]),
                            html.H6("Interpretation:", className="mt-2"),
                            html.Ul([
                                html.Li("Top-right quadrant: High risk (extreme sentiment + high contradiction)"),
                                html.Li("Bottom-left quadrant: Low risk (moderate sentiment + low contradiction)"),
                                html.Li("Large red points: High-priority cases for investigation"),
                                html.Li("Small green points: Lower-risk communications")
                            ])
                        ], className="mb-4"),
                        
                        # Comment Analysis
                        html.Div([
                            html.H5("4. Comment Analysis", className="text-primary"),
                            html.P([
                                html.Strong("Purpose: "),
                                "Evaluate public response and engagement."
                            ]),
                            html.H6("Methodology:", className="mt-2"),
                            html.Ul([
                                html.Li("Sentiment classification of individual comments"),
                                html.Li("Engagement tracking over time"),
                                html.Li([
                                    "Rolling average of engagement:",
                                    html.Code("window=3, min_periods=1, center=True")
                                ]),
                                html.Li("Distribution analysis of comment sentiments")
                            ]),
                            html.H6("Interpretation:", className="mt-2"),
                            html.Ul([
                                html.Li("High negative/skeptical ratio: Potential credibility issues"),
                                html.Li("Engagement spikes: Notable events or controversial posts"),
                                html.Li("Comment sentiment distribution shows public perception"),
                                html.Li("Trend analysis reveals long-term reception patterns")
                            ])
                        ], className="mb-4")
                    ]),
                    id="methodology-collapse",
                    is_open=False,
                )
            ])
        ])
    ], className="mb-4")
])

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
        # Convert string representations of lists back to lists
        df['comments'] = df['comments'].apply(lambda x: x.split('|') if pd.notna(x) and x != '' else [])
        df['comment_sentiments'] = df['comment_sentiments'].apply(lambda x: x.split('|') if pd.notna(x) and x != '' else [])
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
    # Convert string dates to datetime objects
    if start_date:
        start_date = datetime.strptime(start_date.split('T')[0], '%Y-%m-%d')
    if end_date:
        end_date = datetime.strptime(end_date.split('T')[0], '%Y-%m-%d')
    
    # Create a copy and convert timestamp to datetime if it's not already
    filtered_df = df[
        (df['company'] == company) &
        (df['platform'] == platform)
    ].copy()
    
    if not pd.api.types.is_datetime64_any_dtype(filtered_df['timestamp']):
        filtered_df['timestamp'] = pd.to_datetime(filtered_df['timestamp'])
    
    # Apply date filtering if dates are provided
    if start_date:
        filtered_df = filtered_df[filtered_df['timestamp'] >= start_date]
    if end_date:
        filtered_df = filtered_df[filtered_df['timestamp'] <= end_date]
    
    # Sort by timestamp
    filtered_df = filtered_df.sort_values('timestamp')
    
    # Check if we have any data
    if len(filtered_df) == 0:
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
            title='Sentiment Score Trend',
            xaxis_title='Timestamp',
            yaxis_title='Sentiment Score'
        )
        return fig
    
    # Calculate rolling average for smoother trend
    filtered_df['rolling_sentiment'] = filtered_df['sentiment_score'].rolling(
        window=3, min_periods=1, center=True
    ).mean()
    
    # Create the line plot using graph objects
    fig = go.Figure()
    
    # Add the main sentiment trend line
    fig.add_trace(go.Scatter(
        x=filtered_df['timestamp'],
        y=filtered_df['sentiment_score'],
        mode='markers',
        name='Sentiment Score',
        marker=dict(size=8, color='#1f77b4'),
        showlegend=True
    ))
    
    # Add rolling average line
    fig.add_trace(go.Scatter(
        x=filtered_df['timestamp'],
        y=filtered_df['rolling_sentiment'],
        mode='lines',
        name='Trend',
        line=dict(color='#1f77b4', width=2),
        showlegend=True
    ))
    
    # Add a horizontal line at y=0 for reference
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    
    # Update layout
    fig.update_layout(
        title={
            'text': 'Sentiment Score Trend',
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        xaxis_title='Timestamp',
        yaxis_title='Sentiment Score',
        showlegend=True,
        hovermode='x unified',
        yaxis=dict(range=[-1, 1])  # Set y-axis range for sentiment scores
    )
    
    # Update axes
    fig.update_xaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='LightGrey',
        tickformat='%Y-%m-%d'
    )
    fig.update_yaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='LightGrey',
        zeroline=True,
        zerolinewidth=1.5,
        zerolinecolor='gray'
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
    emotion_contexts = []
    for content in filtered_df['content']:
        result = emotion_classifier.get_sustainability_emotions(content)
        if result['has_sustainability_context']:
            # Get all emotions with their scores
            for emotion, score in result['emotions'].items():
                emotions.append(emotion)
                # Add context about greenwashing indicators
                context = []
                if result['greenwashing_indicators'].get('excessive_optimism') and emotion == 'optimism':
                    context.append('High optimism (potential greenwashing)')
                if result['greenwashing_indicators'].get('lack_of_authenticity') and emotion == 'joy':
                    context.append('Excessive joy (potential greenwashing)')
                if result['greenwashing_indicators'].get('overconfidence') and emotion == 'pride':
                    context.append('High pride (potential greenwashing)')
                emotion_contexts.append(' | '.join(context) if context else 'Normal expression')
    
    # Create emotion counts with context
    emotion_data = pd.DataFrame({
        'emotion': emotions,
        'context': emotion_contexts
    })
    
    # Check if we have any data
    if len(emotion_data) == 0:
        fig = go.Figure()
        fig.add_annotation(
            text="No emotion data available for selected filters",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False
        )
        fig.update_layout(
            title='Emotion Distribution'
        )
        return fig
    
    # Get top emotions
    top_emotions = emotion_data['emotion'].value_counts().head(5)
    
    # Create pie chart with custom colors
    fig = px.pie(
        values=top_emotions.values,
        names=top_emotions.index,
        title='Top Emotions Distribution',
        color=top_emotions.index,
        color_discrete_map={
            'optimism': '#2ecc71',  # Green
            'joy': '#f1c40f',      # Yellow
            'pride': '#3498db',    # Blue
            'satisfaction': '#9b59b6',  # Purple
            'approval': '#e67e22',  # Orange
            'neutral': '#95a5a6',   # Gray
            'disapproval': '#e74c3c',  # Red
            'disappointment': '#c0392b',  # Dark Red
            'anger': '#d35400',     # Dark Orange
            'fear': '#8e44ad'       # Dark Purple
        }
    )
    
    # Update layout
    fig.update_layout(
        showlegend=True,
        legend_title_text='Emotion',
        title={
            'text': 'Top Emotions Distribution',
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        }
    )
    
    # Update traces
    fig.update_traces(
        textinfo='percent+label',
        hovertemplate='%{label}<br>Count: %{value}<br>Percentage: %{percent:.1%}<extra></extra>'
    )
    
    return fig

# Callback for comment sentiment distribution
@app.callback(
    Output('comment-sentiment-distribution', 'figure'),
    [Input('company-filter', 'value'),
     Input('platform-filter', 'value')]
)
def update_comment_sentiment_distribution(company, platform):
    filtered_df = df[
        (df['company'] == company) &
        (df['platform'] == platform)
    ]
    
    # Collect all comment sentiments and normalize them
    all_sentiments = []
    for sentiments in filtered_df['comment_sentiments']:
        for sentiment in sentiments:
            # Clean and map sentiment values
            sentiment = sentiment.strip().lower()
            if sentiment.startswith('p'):  # 'p' or 'pos' or 'positive'
                all_sentiments.append('Positive')
            elif sentiment.startswith('s'):  # 's' or 'skeptical'
                all_sentiments.append('Skeptical')
            elif sentiment.startswith('n'):  # 'n' or 'neg' or 'negative'
                all_sentiments.append('Negative')
    
    # Create sentiment counts with proper labels
    sentiment_counts = pd.Series(all_sentiments).value_counts()
    
    # Check if we have any data
    if len(sentiment_counts) == 0:
        fig = go.Figure()
        fig.add_annotation(
            text="No comment data available for selected filters",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False
        )
        fig.update_layout(
            title='Comment Sentiment Distribution'
        )
        return fig
    
    # Create pie chart with custom colors
    fig = px.pie(
        values=sentiment_counts.values,
        names=sentiment_counts.index,
        title='Comment Sentiment Distribution',
        color=sentiment_counts.index,
        color_discrete_map={
            'Positive': '#2ecc71',  # Green
            'Skeptical': '#f1c40f',  # Yellow
            'Negative': '#e74c3c'   # Red
        }
    )
    
    # Update layout
    fig.update_layout(
        showlegend=True,
        legend_title_text='Sentiment',
        title={
            'text': 'Comment Sentiment Distribution',
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        }
    )
    
    # Update traces
    fig.update_traces(
        textinfo='percent+label',
        hovertemplate='%{label}<br>Count: %{value}<br>Percentage: %{percent:.1%}<extra></extra>'
    )
    
    return fig

# Callback for comment engagement trend
@app.callback(
    Output('comment-engagement-trend', 'figure'),
    [Input('company-filter', 'value'),
     Input('platform-filter', 'value'),
     Input('date-range', 'start_date'),
     Input('date-range', 'end_date')]
)
def update_comment_engagement_trend(company, platform, start_date, end_date):
    # Create a copy of the filtered DataFrame
    filtered_df = df[
        (df['company'] == company) &
        (df['platform'] == platform)
    ].copy()
    
    # Convert timestamp to datetime if it's not already
    if not pd.api.types.is_datetime64_any_dtype(filtered_df['timestamp']):
        filtered_df['timestamp'] = pd.to_datetime(filtered_df['timestamp'])
    
    # Convert string dates to datetime objects
    if start_date:
        start_date = datetime.strptime(start_date.split('T')[0], '%Y-%m-%d')
        filtered_df = filtered_df[filtered_df['timestamp'] >= start_date]
    if end_date:
        end_date = datetime.strptime(end_date.split('T')[0], '%Y-%m-%d')
        filtered_df = filtered_df[filtered_df['timestamp'] <= end_date]
    
    # Sort by timestamp
    filtered_df = filtered_df.sort_values('timestamp')
    
    # Calculate number of comments per post using .loc
    filtered_df.loc[:, 'num_comments'] = filtered_df['comments'].apply(len)
    
    # Calculate rolling average for smoother trend
    filtered_df['rolling_comments'] = filtered_df['num_comments'].rolling(
        window=3, min_periods=1, center=True
    ).mean()
    
    # Create the line plot using graph objects
    fig = go.Figure()
    
    # Add scatter plot for actual values
    fig.add_trace(go.Scatter(
        x=filtered_df['timestamp'],
        y=filtered_df['num_comments'],
        mode='markers',
        name='Comments',
        marker=dict(size=8, color='#2ecc71'),
        showlegend=True
    ))
    
    # Add line plot for rolling average
    fig.add_trace(go.Scatter(
        x=filtered_df['timestamp'],
        y=filtered_df['rolling_comments'],
        mode='lines',
        name='Trend',
        line=dict(color='#2ecc71', width=2),
        showlegend=True
    ))
    
    # Update layout
    fig.update_layout(
        title={
            'text': 'Comment Engagement Trend',
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        xaxis_title='Timestamp',
        yaxis_title='Number of Comments',
        showlegend=True,
        hovermode='x unified'
    )
    
    # Update axes
    fig.update_xaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='LightGrey',
        tickformat='%Y-%m-%d'
    )
    fig.update_yaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='LightGrey',
        rangemode='nonnegative'  # Ensure y-axis starts at 0 or the minimum value
    )
    
    return fig

# Callback for comment summary
@app.callback(
    Output('comment-summary', 'children'),
    [Input('company-filter', 'value'),
     Input('platform-filter', 'value')]
)
def update_comment_summary(company, platform):
    filtered_df = df[
        (df['company'] == company) &
        (df['platform'] == platform)
    ]
    
    # Calculate comment statistics
    total_comments = sum(len(comments) for comments in filtered_df['comments'])
    sentiment_counts = {'positive': 0, 'skeptical': 0, 'negative': 0}
    
    # Process comment sentiments
    for sentiments in filtered_df['comment_sentiments']:
        for sentiment in sentiments:
            # Clean and validate sentiment value
            sentiment = sentiment.strip().lower()
            if sentiment in sentiment_counts:
                sentiment_counts[sentiment] += 1
            else:
                # Handle invalid sentiments by mapping to closest category
                if sentiment.startswith('p'):  # 'p' or 'pos' or 'positive'
                    sentiment_counts['positive'] += 1
                elif sentiment.startswith('s'):  # 's' or 'skeptical'
                    sentiment_counts['skeptical'] += 1
                elif sentiment.startswith('n'):  # 'n' or 'neg' or 'negative'
                    sentiment_counts['negative'] += 1
    
    return html.Div([
        html.H5("Comment Statistics"),
        html.Ul([
            html.Li(f"Total Comments: {total_comments}"),
            html.Li(f"Positive Comments: {sentiment_counts['positive']}"),
            html.Li(f"Skeptical Comments: {sentiment_counts['skeptical']}"),
            html.Li(f"Negative Comments: {sentiment_counts['negative']}")
        ])
    ])

# Callback for greenwashing risk
@app.callback(
    Output('greenwashing-risk', 'figure'),
    [Input('company-filter', 'value'),
     Input('platform-filter', 'value')]
)
def update_greenwashing_risk(company, platform):
    # Create a copy of the filtered DataFrame
    filtered_df = df[
        (df['company'] == company) &
        (df['platform'] == platform)
    ].copy()
    
    # Check if we have any data
    if len(filtered_df) == 0:
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
    for idx, row in filtered_df.iterrows():
        sentiment = row['sentiment_score']
        contradiction = row['contradiction_score']
        
        # Calculate comment-based risk
        comment_risk = 0
        if row['comments']:
            negative_comments = sum(1 for s in row['comment_sentiments'] if s == 'negative')
            skeptical_comments = sum(1 for s in row['comment_sentiments'] if s == 'skeptical')
            comment_risk = (negative_comments + 0.5 * skeptical_comments) / len(row['comments'])
        
        # Check for NaN values
        if pd.isna(sentiment) or pd.isna(contradiction):
            continue
            
        # Combine post and comment risk scores
        risk_score = (abs(sentiment) + contradiction + comment_risk) / 3
        risk_scores.append(risk_score)
    
    # Check if we have any valid risk scores
    if not risk_scores:
        fig = go.Figure()
        fig.add_annotation(
            text="No valid data points for risk calculation",
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
    
    # Convert risk scores to numpy array and normalize for better visualization
    risk_scores = np.array(risk_scores)
    # Remove any remaining NaN values
    risk_scores = risk_scores[~np.isnan(risk_scores)]
    
    if len(risk_scores) == 0:
        fig = go.Figure()
        fig.add_annotation(
            text="No valid data points for risk calculation",
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
    
    # Normalize sizes between 5 and 25
    normalized_sizes = (risk_scores - risk_scores.min()) / (risk_scores.max() - risk_scores.min()) * 20 + 5
    
    # Create scatter plot with valid data only
    valid_df = filtered_df[~filtered_df['sentiment_score'].isna() & ~filtered_df['contradiction_score'].isna()].copy()
    
    # Create the scatter plot using graph objects
    fig = go.Figure()
    
    # Add scatter plot with hover text
    fig.add_trace(go.Scatter(
        x=valid_df['sentiment_score'],
        y=valid_df['contradiction_score'],
        mode='markers',  # Remove text mode, only show markers
        marker=dict(
            size=normalized_sizes,
            color=risk_scores,
            colorscale='RdYlGn_r',  # Red to Green color scale (red for high risk)
            showscale=True,
            colorbar=dict(
                title='Risk Score',
                titleside='right',
                ticktext=['High Risk', 'Medium Risk', 'Low Risk'],
                tickvals=[risk_scores.max(), risk_scores.mean(), risk_scores.min()]
            )
        ),
        name='Posts',
        hovertemplate=(
            "<b>Post ID: %{customdata}</b><br>" +
            "Sentiment Score: %{x:.2f}<br>" +
            "Contradiction Score: %{y:.2f}<br>" +
            "Risk Score: %{marker.color:.2f}<br>" +
            "<b>Content:</b> %{text}<br>" +
            "<extra></extra>"
        ),
        customdata=valid_df['post_id'],  # Add post ID for hover
        text=valid_df['content'].str[:100] + '...'  # Show truncated content in hover
    ))
    
    # Add quadrant labels instead of arrows
    fig.add_annotation(
        text="High Risk Zone<br>(High Contradiction,<br>Extreme Sentiment)",
        x=0.8,
        y=0.8,
        xref="paper",
        yref="paper",
        showarrow=False,
        font=dict(size=10),
        align="center",
        bgcolor="rgba(255, 255, 255, 0.8)"
    )
    
    fig.add_annotation(
        text="Low Risk Zone<br>(Low Contradiction,<br>Moderate Sentiment)",
        x=0.2,
        y=0.2,
        xref="paper",
        yref="paper",
        showarrow=False,
        font=dict(size=10),
        align="center",
        bgcolor="rgba(255, 255, 255, 0.8)"
    )
    
    # Update layout
    fig.update_layout(
        title={
            'text': 'Greenwashing Risk Analysis',
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        xaxis_title='Sentiment Score (Higher = More Positive)',
        yaxis_title='Contradiction Score (Higher = More Contradictory)',
        showlegend=True,
        hovermode='closest'
    )
    
    # Update axes
    fig.update_xaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='LightGrey',
        range=[-1, 1],  # Set x-axis range for sentiment scores
        zeroline=True,
        zerolinewidth=1,
        zerolinecolor='black'
    )
    fig.update_yaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='LightGrey',
        range=[0, 1],  # Set y-axis range for contradiction scores
    )
    
    # Add explanation text
    fig.add_annotation(
        text=(
            "• Points represent individual posts<br>" +
            "• Size: Risk level (larger = higher risk)<br>" +
            "• Color: Risk score (red = high, green = low)<br>" +
            "• Hover over points to see post details"
        ),
        x=0.02,
        y=0.98,
        xref="paper",
        yref="paper",
        showarrow=False,
        align="left",
        bgcolor="rgba(255, 255, 255, 0.8)",
        font=dict(size=10)
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
    
    # Calculate comment metrics
    total_comments = sum(len(comments) for comments in filtered_df['comments'])
    sentiment_counts = {'positive': 0, 'skeptical': 0, 'negative': 0}
    
    # Process comment sentiments
    for sentiments in filtered_df['comment_sentiments']:
        for sentiment in sentiments:
            # Clean and validate sentiment value
            sentiment = sentiment.strip().lower()
            if sentiment in sentiment_counts:
                sentiment_counts[sentiment] += 1
            else:
                # Handle invalid sentiments by mapping to closest category
                if sentiment.startswith('p'):  # 'p' or 'pos' or 'positive'
                    sentiment_counts['positive'] += 1
                elif sentiment.startswith('s'):  # 's' or 'skeptical'
                    sentiment_counts['skeptical'] += 1
                elif sentiment.startswith('n'):  # 'n' or 'neg' or 'negative'
                    sentiment_counts['negative'] += 1
    
    # Analyze emotions and greenwashing indicators
    emotions = []
    greenwashing_indicators = {
        'excessive_optimism': 0,
        'lack_of_authenticity': 0,
        'overconfidence': 0
    }
    
    for content in filtered_df['content']:
        result = emotion_classifier.get_sustainability_emotions(content)
        if result['has_sustainability_context']:
            # Collect all emotions
            emotions.extend(result['emotions'].keys())
            
            # Count greenwashing indicators
            indicators = result['greenwashing_indicators']
            if indicators.get('excessive_optimism'):
                greenwashing_indicators['excessive_optimism'] += 1
            if indicators.get('lack_of_authenticity'):
                greenwashing_indicators['lack_of_authenticity'] += 1
            if indicators.get('overconfidence'):
                greenwashing_indicators['overconfidence'] += 1
    
    # Get top emotions with counts
    top_emotions = pd.Series(emotions).value_counts().head(3)
    
    # Create comment analysis section with percentage calculations only if there are comments
    comment_analysis = []
    if total_comments > 0:
        comment_analysis = [
            html.Li(f"Total Comments: {total_comments}"),
            html.Li(f"Positive Comments: {sentiment_counts['positive']} ({sentiment_counts['positive']/total_comments*100:.1f}%)"),
            html.Li(f"Skeptical Comments: {sentiment_counts['skeptical']} ({sentiment_counts['skeptical']/total_comments*100:.1f}%)"),
            html.Li(f"Negative Comments: {sentiment_counts['negative']} ({sentiment_counts['negative']/total_comments*100:.1f}%)")
        ]
    else:
        comment_analysis = [
            html.Li("No comments available for the selected filters")
        ]
    
    # Create emotion analysis section
    emotion_analysis = []
    if top_emotions.size > 0:
        emotion_analysis = [
            html.Li(f"{emotion}: {count} occurrences")
            for emotion, count in top_emotions.items()
        ]
    else:
        emotion_analysis = [
            html.Li("No emotion data available for the selected filters")
        ]
    
    # Create greenwashing indicators section
    greenwashing_analysis = []
    if any(count > 0 for count in greenwashing_indicators.values()):
        greenwashing_analysis = [
            html.Li(f"Posts with excessive optimism: {greenwashing_indicators['excessive_optimism']}"),
            html.Li(f"Posts with lack of authenticity: {greenwashing_indicators['lack_of_authenticity']}"),
            html.Li(f"Posts with overconfidence: {greenwashing_indicators['overconfidence']}")
        ]
    else:
        greenwashing_analysis = [
            html.Li("No significant greenwashing indicators detected")
        ]
    
    return html.Div([
        html.H5("Key Metrics"),
        html.Ul([
            html.Li(f"Average Sentiment Score: {avg_sentiment:.2f}"),
            html.Li(f"Average Contradiction Score: {avg_contradiction:.2f}"),
            html.Li(f"Total Posts Analyzed: {total_posts}")
        ]),
        html.H5("Comment Analysis"),
        html.Ul(comment_analysis),
        html.H5("Top Emotions"),
        html.Ul(emotion_analysis),
        html.H5("Greenwashing Indicators"),
        html.Ul(greenwashing_analysis)
    ])

# Add callback for methodology collapse
@app.callback(
    Output("methodology-collapse", "is_open"),
    [Input("methodology-collapse-button", "n_clicks")],
    [State("methodology-collapse", "is_open")],
)
def toggle_methodology_collapse(n_clicks, is_open):
    if n_clicks:
        return not is_open
    return is_open

if __name__ == '__main__':
    app.run(debug=True) 