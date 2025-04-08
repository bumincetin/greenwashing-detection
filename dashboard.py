import dash
from dash import html, dcc, Input, Output, State
import dash_bootstrap_components as dbc
from dash_bootstrap_components import NavbarSimple, NavItem, NavLink
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
from sentiment_analyzer import SentimentAnalyzer
from emotion_classifier import EmotionClassifier
from contradiction_detector import ContradictionDetector
from data_generator import DataGenerator
import base64
import io
import json
import os
from dash.exceptions import PreventUpdate

# Set default plotly template
pio.templates.default = "plotly_white"

# Initialize the Dash app with a modern theme
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Custom CSS for better styling
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            .navbar-brand {
                font-size: 1.5rem;
                font-weight: 600;
            }
            .card {
                border: none;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                transition: transform 0.2s;
            }
            .card:hover {
                transform: translateY(-2px);
            }
            .card-header {
                background-color: #f8f9fa;
                border-bottom: 1px solid rgba(0,0,0,0.1);
                font-weight: 600;
            }
            .upload-box {
                border: 2px dashed #6c757d;
                border-radius: 8px;
                padding: 20px;
                text-align: center;
                background-color: #f8f9fa;
                transition: all 0.3s;
            }
            .upload-box:hover {
                border-color: #0d6efd;
                background-color: #e9ecef;
            }
            .badge {
                font-size: 0.8rem;
                padding: 0.4em 0.8em;
            }
            .text-info {
                color: #0dcaf0 !important;
            }
            .text-danger {
                color: #dc3545 !important;
            }
            .footer {
                background-color: #f8f9fa;
                padding: 20px 0;
                margin-top: 40px;
            }
            .graph-container {
                background-color: white;
                border-radius: 8px;
                padding: 15px;
                margin-bottom: 20px;
            }
            .filter-section {
                background-color: #f8f9fa;
                border-radius: 8px;
                padding: 20px;
                margin-bottom: 30px;
            }
            .metric-card {
                background: linear-gradient(45deg, #0d6efd, #0dcaf0);
                color: white;
                border-radius: 8px;
                padding: 20px;
                margin-bottom: 20px;
            }
            .metric-value {
                font-size: 2rem;
                font-weight: 600;
            }
            .metric-label {
                font-size: 0.9rem;
                opacity: 0.9;
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

# Initialize analyzers and global variables
sentiment_analyzer = SentimentAnalyzer()
emotion_classifier = EmotionClassifier()
contradiction_detector = ContradictionDetector()
data_generator = DataGenerator()

# Initialize global DataFrame
global df
df = None

# Generate synthetic data
print("Generating synthetic data...")
df = data_generator.generate_synthetic_data(num_posts=500)  # Generate 500 posts total

# Standardize platform names
def standardize_platform(platform):
    platform = platform.lower()
    if 'linkedin' in platform:
        return 'LinkedIn'
    elif 'twitter' in platform or 'x' in platform:
        return 'Twitter'
    elif 'facebook' in platform:
        return 'Facebook'
    elif 'instagram' in platform:
        return 'Instagram'
    else:
        return 'Other'

df['platform_standardized'] = df['platform'].apply(standardize_platform)

# Save the synthetic data to CSV
output_file = 'synthetic_sustainability_data.csv'
df.to_csv(output_file, index=False)
print(f"Generated {len(df)} rows of synthetic data")
print(f"Companies: {df['company'].nunique()}")
print(f"Platforms: {df['platform_standardized'].unique()}")
print(f"Data saved to {output_file}")

# Define the layout
app.layout = html.Div([
    # Navbar with gradient background
    dbc.NavbarSimple(
        children=[
            html.Span("Sustainability Claims Dashboard", className="navbar-brand text-white")
        ],
        color="primary",
        dark=True,
        className="mb-4 shadow-sm",
        style={
            'background': 'linear-gradient(45deg, #0d6efd, #0dcaf0)',
            'padding': '1rem 2rem'
        }
    ),
    
    dbc.Container([
        # Documentation and Data Format Card with improved styling
        dcc.Store(id='store-data', storage_type='memory'),
        
        # Methodology Section
        dbc.Card([
            dbc.CardHeader("Methodology", className="bg-primary text-white"),
            dbc.CardBody([
                dbc.Accordion([
                    dbc.AccordionItem([
                        html.Div([
                            html.H5("Sentiment Analysis", className="text-primary"),
                            html.P("We analyze the sentiment of sustainability claims using:"),
                            html.Ul([
                                html.Li("Content analysis of posts and comments"),
                                html.Li("Sentiment scores ranging from -1 (negative) to 1 (positive)"),
                                html.Li("Rolling averages to identify trends"),
                                html.Li("Platform-specific sentiment patterns")
                            ])
                        ])
                    ], title="Sentiment Analysis"),
                    
                    dbc.AccordionItem([
                        html.Div([
                            html.H5("Greenwashing Risk Assessment", className="text-primary"),
                            html.P("Risk scores are calculated based on three main factors:"),
                            html.Ul([
                                html.Li("Sentiment Score: Measures the emotional intensity of claims"),
                                html.Li("Contradiction Score: Identifies inconsistencies in messaging"),
                                html.Li("Comment Analysis: Evaluates public response and skepticism"),
                                html.P("The final risk score is calculated as: (|sentiment| + contradiction + comment_risk) / 3")
                            ])
                        ])
                    ], title="Risk Assessment"),
                    
                    dbc.AccordionItem([
                        html.Div([
                            html.H5("Emotion Analysis", className="text-primary"),
                            html.P("We identify emotions in sustainability claims using:"),
                            html.Ul([
                                html.Li("Keyword-based emotion detection"),
                                html.Li("Context-aware sentiment analysis"),
                                html.Li("Pattern recognition in sustainability discourse"),
                                html.Li("Emotion distribution across platforms")
                            ])
                        ])
                    ], title="Emotion Analysis"),
                    
                    dbc.AccordionItem([
                        html.Div([
                            html.H5("Comment Analysis", className="text-primary"),
                            html.P("Comment analysis includes:"),
                            html.Ul([
                                html.Li("Sentiment categorization (positive, skeptical, negative)"),
                                html.Li("Engagement metrics tracking"),
                                html.Li("Temporal analysis of public response"),
                                html.Li("Contradiction detection in discussions")
                            ])
                        ])
                    ], title="Comment Analysis")
                ], start_collapsed=True, className="mb-3")
            ])
        ], className='mb-4 shadow-sm'),
        
        # Data Format and Structure section
        html.Div(id='data-format-section', className='mt-4 p-4 bg-light rounded shadow-sm', children=[
            html.H5("Required Data Format", className="text-primary mb-3"),
            html.P("Upload a CSV file with the following columns:", className="text-muted"),
            html.Ul([
                html.Li([html.Code("post_id"), ": Unique identifier (integer)"], className="mb-2"),
                html.Li([html.Code("company"), ": Company name (string)"], className="mb-2"),
                html.Li([html.Code("platform"), ": Social media platform (string)"], className="mb-2"),
                html.Li([html.Code("timestamp"), ": Timestamp (YYYY-MM-DD HH:MM:SS)"], className="mb-2"),
                html.Li([html.Code("content"), ": Post text (string)"], className="mb-2"),
                html.Li([html.Code("engagement_metrics"), ": JSON string of metrics"], className="mb-2"),
                html.Li([html.Code("comments"), ": JSON array of comment strings"], className="mb-2")
            ], className="list-unstyled"),
            
            html.H5("Data Format Options", className="text-primary mt-4 mb-3"),
            dbc.Row([
                dbc.Col([
                    dcc.Dropdown(
                        id='data-format-dropdown',
                        options=[
                            {'label': 'Standard Format', 'value': 'standard'},
                            {'label': 'Simple Format (No JSON)', 'value': 'simple'},
                            {'label': 'Extended Format (With Sentiment)', 'value': 'extended'},
                            {'label': 'Minimal Format', 'value': 'minimal'}
                        ],
                        value='standard',
                        clearable=False,
                        className="mb-3"
                    ),
                ], width=12),
            ]),
            html.Div(id='format-description', className="mt-3")
        ]),
        
        # File Upload with improved styling
        dbc.Card([
            dbc.CardHeader("Upload Data", className="bg-primary text-white"),
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        dcc.Upload(
                            id='upload-data',
                            children=html.Div([
                                html.I(className="fas fa-cloud-upload-alt fa-2x mb-2"),
                                html.Div('Drag and Drop or Click to Select a CSV File', className="upload-text"),
                                html.Div('Accepted format: CSV files only', className="text-muted small mt-1")
                            ], className='upload-box'),
                            style={
                                'width': '100%',
                                'height': '150px',
                                'lineHeight': '60px',
                                'borderWidth': '2px',
                                'borderStyle': 'dashed',
                                'borderRadius': '8px',
                                'textAlign': 'center',
                                'backgroundColor': '#f8f9fa',
                                'cursor': 'pointer'
                            },
                            multiple=False,
                            accept='.csv'
                        ),
                        # Add loading spinner and status
                        dbc.Spinner(
                            html.Div(id='output-data-upload', className="mt-3"),
                            color="primary",
                            type="grow",
                            fullscreen=False
                        ),
                        html.Div(id='upload-status', className='mt-2 text-center')
                    ], width=12),
                ]),
            ]),
        ], className='mb-4 shadow-sm'),
        
        # Filters Section with improved layout
        dbc.Card([
            dbc.CardHeader("Filters", className="bg-primary text-white"),
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        html.Label("Company", className="fw-bold mb-2"),
                        dcc.Dropdown(
                            id='company-filter',
                            options=[],
                            placeholder="Select a company",
                            className="mb-3"
                        ),
                        html.Div(id='company-risk-container', className='mt-2 d-flex justify-content-start align-items-center bg-light p-3 rounded', children=[
                            html.Span("Greenwashing Risk Score: ", className='fw-bold me-2'),
                            html.Span(id='company-risk-score', className='me-2 fw-bold text-primary'),
                            html.Span(id='company-risk-description')
                        ])
                    ], width=6),
                    dbc.Col([
                        html.Label("Platform", className="fw-bold mb-2"),
                        dcc.Dropdown(
                            id='platform-filter',
                            options=[],
                            placeholder="Select a platform",
                            className="mb-3"
                        ),
                        html.Label("Date Range", className='fw-bold mb-2'),
                        dcc.DatePickerRange(
                            id='date-range',
                            start_date_placeholder_text="Start Date",
                            end_date_placeholder_text="End Date",
                            calendar_orientation='horizontal',
                            clearable=True,
                            with_portal=True,
                            min_date_allowed=date(2010, 1, 1),
                            max_date_allowed=date(2030, 12, 31),
                            className="w-100"
                        ),
                    ], width=6),
                ]),
            ]),
        ], className='mb-4 shadow-sm'),
        
        # Main Analysis Section with improved layout
        dbc.Row([
            # Sentiment Analysis Column
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Sentiment Analysis", className="bg-primary text-white"),
                    dbc.CardBody([
                        html.Div(className="graph-container", children=[
                            dcc.Graph(id='sentiment-trend')
                        ]),
                        html.Hr(),
                        html.Div(className="graph-container", children=[
                            dcc.Graph(id='sentiment-by-platform')
                        ])
                    ])
                ], className='h-100 shadow-sm')
            ], width=6),
            
            # Greenwashing Risk Column
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Greenwashing Risk Analysis", className="bg-primary text-white"),
                    dbc.CardBody([
                        html.Div(className="graph-container", children=[
                            dcc.Graph(id='greenwashing-viz')
                        ]),
                        html.Hr(),
                        html.Div(id='top-contradictions', className="mt-3")
                    ])
                ], className='h-100 shadow-sm')
            ], width=6)
        ], className='mb-4'),
        
        # Detailed Analysis Section with improved layout
        dbc.Card([
            dbc.CardHeader("Detailed Analysis", className="bg-primary text-white"),
            dbc.CardBody([
                html.Div(id='detailed-analysis', className="mb-4"),
                html.Hr(),
                dbc.Row([
                    dbc.Col([
                        html.H5("Emotion Analysis", className="text-primary mb-3"),
                        html.Div(className="graph-container", children=[
                            dcc.Graph(id='emotion-distribution')
                        ])
                    ], width=6),
                    dbc.Col([
                        html.H5("Comment Analysis", className="text-primary mb-3"),
                        html.Div(id='comment-summary', className="mb-3"),
                        html.Div(className="graph-container", children=[
                            dcc.Graph(id='comment-sentiment-distribution')
                        ]),
                        html.Div(className="graph-container", children=[
                            dcc.Graph(id='comment-engagement-trend')
                        ])
                    ], width=6)
                ])
            ])
        ], className='mb-4 shadow-sm'),
        
        # Posts List Section with improved styling
        dbc.Card([
            dbc.CardHeader("Posts Analysis", className="bg-primary text-white"),
            dbc.CardBody([
                html.Div(id='posts-list', className="posts-container")
            ])
        ], className='mb-4 shadow-sm'),
        
        # Footer with improved styling
        html.Div([
            html.P("© 2023 Sustainability Claims Dashboard - Created with Dash", 
                  className="text-center mt-5 mb-3 text-muted")
        ], className="footer")
    ], fluid=True)
])

# Callback for methodology collapse (Removed as Accordion handles this)

# Callback for data upload and processing
@app.callback(
    [Output('upload-status', 'children'),
     Output('company-filter', 'options'),
     Output('platform-filter', 'options'),
     Output('upload-status', 'className'),
     Output('store-data', 'data')],
    [Input('upload-data', 'contents')],
    [State('upload-data', 'filename')]
)
def update_output(contents, filename):
    global df
    
    if contents is None:
        return "No file uploaded yet", [], [], 'mt-2 text-center text-muted', None
    
    try:
        # Validate file type
        if not filename.endswith('.csv'):
            return "Error: Please upload a CSV file", [], [], 'mt-2 text-center text-danger', None
        
        # Parse the uploaded content
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        
        # Try to read the CSV data
        try:
            temp_df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
        except UnicodeDecodeError:
            # Try with different encoding if UTF-8 fails
            temp_df = pd.read_csv(io.StringIO(decoded.decode('latin-1')))
        
        # Validate required columns
        required_columns = ['company', 'platform', 'timestamp', 'content']
        missing_columns = [col for col in required_columns if col not in temp_df.columns]
        
        if missing_columns:
            return (
                f"Error: Missing required columns: {', '.join(missing_columns)}",
                [], [], 'mt-2 text-center text-danger', None
            )
        
        # Process JSON columns if they exist
        if 'comments' in temp_df.columns:
            temp_df['comments'] = temp_df['comments'].apply(
                lambda x: json.loads(x) if pd.notna(x) and isinstance(x, str) else []
            )
        
        if 'engagement_metrics' in temp_df.columns:
            temp_df['engagement_metrics'] = temp_df['engagement_metrics'].apply(
                lambda x: json.loads(x) if pd.notna(x) and isinstance(x, str) else {}
            )
        
        # Add standardized platform
        temp_df['platform_standardized'] = temp_df['platform'].apply(standardize_platform)
        
        # Calculate sentiment scores if not present
        if 'sentiment_score' not in temp_df.columns:
            temp_df['sentiment_score'] = temp_df['content'].apply(
                lambda x: sentiment_analyzer.analyze_sentiment(x) if pd.notna(x) else 0
            )
        
        # Calculate contradiction scores if not present
        if 'contradiction_score' not in temp_df.columns:
            temp_df['contradiction_score'] = temp_df.apply(
                lambda row: contradiction_detector.detect_contradiction(
                    row['content'], row.get('comments', [])
                ) if pd.notna(row['content']) else 0,
                axis=1
            )
        
        # Update the global dataframe
        df = temp_df
        
        # Store the data in JSON format for the dcc.Store
        stored_data = df.to_dict('records')
        
        # Return success message, dropdown options, and stored data
        return (
            html.Div([
                html.I(className="fas fa-check-circle text-success me-2"),
                f"Successfully uploaded {filename} ({len(df)} rows)"
            ]),
            [{'label': company, 'value': company} for company in sorted(df['company'].unique())],
            [{'label': platform, 'value': platform} for platform in sorted(df['platform_standardized'].unique())],
            'mt-2 text-center text-success',
            stored_data
        )
        
    except Exception as e:
        error_message = str(e)
        if len(error_message) > 100:
            error_message = error_message[:100] + "..."
        return f"Error: {error_message}", [], [], 'mt-2 text-center text-danger', None

# Helper Function for Filtering
def filter_data(company, platform, start_date, end_date):
    """Filter the DataFrame based on the selected criteria"""
    global df
    
    if df is None or df.empty:
        return pd.DataFrame()  # Return empty DataFrame if no data
        
    filtered = df.copy()
    
    # Apply filters only if they are provided
    if company:
        filtered = filtered[filtered['company'] == company]
    
    if platform:
        filtered = filtered[filtered['platform_standardized'] == platform]
    
    # Convert timestamp column to datetime only if needed for filtering
    if (start_date or end_date) and not pd.api.types.is_datetime64_any_dtype(filtered['timestamp']):
        filtered['timestamp'] = pd.to_datetime(filtered['timestamp'])
        
    if start_date:
        start_dt = datetime.strptime(start_date.split('T')[0], '%Y-%m-%d')
        filtered = filtered[filtered['timestamp'] >= start_dt]
    
    if end_date:
        end_dt = datetime.strptime(end_date.split('T')[0], '%Y-%m-%d')
        filtered = filtered[filtered['timestamp'] <= end_dt]
    
    return filtered.sort_values('timestamp')

# --- Callback Definitions --- 

# Callback for sentiment trend
@app.callback(
    Output('sentiment-trend', 'figure'),
    [Input('company-filter', 'value'),
     Input('platform-filter', 'value'),
     Input('date-range', 'start_date'),
     Input('date-range', 'end_date')]
)
def update_sentiment_trend(company, platform, start_date, end_date):
    filtered_df = filter_data(company, platform, start_date, end_date)
    
    if filtered_df.empty:
        fig = go.Figure()
        fig.update_layout(
            title='Sentiment Trend Over Time',
            xaxis={'visible': False},
            yaxis={'visible': False},
            annotations=[{'text': 'No data available for selected filters', 'showarrow': False, 'font': {'size': 20}}]
        )
        return fig
    
    # Calculate rolling average for sentiment
    filtered_df = filtered_df.sort_values('timestamp')
    filtered_df['sentiment_rolling'] = filtered_df['sentiment_score'].rolling(window=3, min_periods=1, center=True).mean()
    
    # Create figure with both actual and rolling average sentiment
    fig = px.line(
        filtered_df,
        x='timestamp',
        y=['sentiment_score', 'sentiment_rolling'],
        title=f'Sentiment Trend Over Time for {company} on {platform if platform else "All Platforms"}',
        labels={'value': 'Sentiment Score (-1 to 1)', 'timestamp': 'Date', 'variable': 'Metric'}
    )
    
    # Update line names and styling
    fig.data[0].name = 'Individual Posts'
    fig.data[0].line.width = 1
    fig.data[0].line.color = 'rgba(0,123,255,0.5)'
    
    fig.data[1].name = 'Rolling Average (3 posts)'
    fig.data[1].line.width = 3
    fig.data[1].line.color = 'rgba(0,123,255,1)'
    
    # Add horizontal reference line at y=0
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    
    # Add annotation zones for positive/negative sentiment
    fig.add_annotation(x=0.02, y=0.85, xref="paper", yref="paper",
                      text="Positive Sentiment", showarrow=False,
                      font=dict(size=12, color="green"),
                      align="left")
    
    fig.add_annotation(x=0.02, y=0.15, xref="paper", yref="paper",
                      text="Negative Sentiment", showarrow=False,
                      font=dict(size=12, color="red"),
                      align="left")
    
    # Update layout
    fig.update_layout(
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=20, r=20, t=40, b=20),
        hovermode="closest"
    )
    
    # Update y-axis range slightly beyond data range for clarity
    y_min = min(filtered_df['sentiment_score'].min(), -0.1) - 0.1
    y_max = max(filtered_df['sentiment_score'].max(), 0.1) + 0.1
    
    fig.update_yaxes(range=[y_min, y_max])
    
    fig.update_traces(hovertemplate='Date: %{x}<br>Score: %{y:.2f}')
    
    return fig

# Callback for sentiment by platform
@app.callback(
    Output('sentiment-by-platform', 'figure'),
    [Input('company-filter', 'value'),
     Input('date-range', 'start_date'),
     Input('date-range', 'end_date')]
)
def update_sentiment_by_platform(company, start_date, end_date):
    # Filter by company and date, but not by platform (we want to compare across platforms)
    filtered_df = filter_data(company, None, start_date, end_date)
    
    if filtered_df.empty:
        fig = go.Figure()
        fig.update_layout(
            title='Sentiment Analysis by Platform',
            xaxis={'visible': False},
            yaxis={'visible': False},
            annotations=[{'text': 'No data available for selected filters', 'showarrow': False, 'font': {'size': 20}}]
        )
        return fig
    
    # Group by platform
    platform_sentiment = filtered_df.groupby('platform_standardized')['sentiment_score'].agg(
        ['mean', 'count', 'std']
    ).reset_index()
    
    platform_sentiment.columns = ['platform', 'avg_sentiment', 'post_count', 'sentiment_std']
    
    # Sort by post count
    platform_sentiment = platform_sentiment.sort_values(by='post_count', ascending=False)
    
    # Set a minimum post count for reliability
    platform_sentiment = platform_sentiment[platform_sentiment['post_count'] >= 3]
    
    if platform_sentiment.empty:
        fig = go.Figure()
        fig.update_layout(
            title='Sentiment Analysis by Platform',
            xaxis={'visible': False},
            yaxis={'visible': False},
            annotations=[{'text': 'Not enough data by platform for comparison', 'showarrow': False, 'font': {'size': 20}}]
        )
        return fig
    
    # Define colors
    sentiment_colors = []
    for sentiment in platform_sentiment['avg_sentiment']:
        if sentiment > 0.3:
            color = 'green'
        elif sentiment > 0:
            color = 'lightgreen'
        elif sentiment > -0.3:
            color = 'lightsalmon'
        else:
            color = 'red'
        sentiment_colors.append(color)
    
    # Create figure with bars for average sentiment
    fig = go.Figure()
    
    # Add sentiment bars with error bars
    fig.add_trace(go.Bar(
        x=platform_sentiment['platform'],
        y=platform_sentiment['avg_sentiment'],
        marker_color=sentiment_colors,
        text=[f"Posts: {count}" for count in platform_sentiment['post_count']],
        textposition='auto',
        error_y=dict(
            type='data',
            array=platform_sentiment['sentiment_std'],
            visible=True
        ),
        hovertemplate='Platform: %{x}<br>Average Sentiment: %{y:.2f}<br>Posts: %{text}<extra></extra>'
    ))
    
    # Add horizontal reference line at y=0
    fig.add_shape(type="line",
                 x0=-0.5,
                 y0=0,
                 x1=len(platform_sentiment) - 0.5,
                 y1=0,
                 line=dict(color="gray", width=1, dash="dash"))
    
    # Update layout
    fig.update_layout(
        title=f"Sentiment Analysis by Platform for {company}",
        xaxis_title='Platform',
        yaxis_title='Average Sentiment Score',
        yaxis=dict(range=[-1, 1]),
        margin=dict(l=20, r=20, t=40, b=80),  # Larger bottom margin for rotated labels
    )
    
    # Rotate x-axis labels for better readability
    fig.update_xaxes(tickangle=45)
    
    return fig

# Callback for emotion distribution
@app.callback(
    Output('emotion-distribution', 'figure'),
    [Input('company-filter', 'value'),
     Input('platform-filter', 'value'),
     Input('date-range', 'start_date'),
     Input('date-range', 'end_date')]
)
def update_emotion_distribution(company, platform, start_date, end_date):
    filtered_df = filter_data(company, platform, start_date, end_date)
    
    if filtered_df.empty:
         return go.Figure(layout={
            'title': 'Emotion Distribution in Posts',
            'xaxis': {'visible': False}, 'yaxis': {'visible': False},
            'annotations': [{'text': 'No data for selected filters', 'showarrow': False}]
        })
        
    emotions_list = []
    for content in filtered_df['content']:
        if pd.notna(content):
             # Use the emotion classifier
            emotions = emotion_classifier.get_sustainability_emotions(content)
            if emotions.get('has_sustainability_context', False):
                emotions_list.extend(emotions.get('emotions', {}).keys())
                
    if not emotions_list:
         return go.Figure(layout={
            'title': 'Emotion Distribution in Posts',
            'xaxis': {'visible': False}, 'yaxis': {'visible': False},
            'annotations': [{'text': 'No sustainability-related emotions detected', 'showarrow': False}]
        })

    emotion_counts = pd.Series(emotions_list).value_counts().reset_index()
    emotion_counts.columns = ['Emotion', 'Count']
    
    fig = px.bar(
        emotion_counts,
        x='Emotion',
        y='Count',
        title='Emotion Distribution in Posts'
        # template='plotly_white' # Set globally now
    )
    fig.update_layout(xaxis_title="Emotion Category", yaxis_title="Number of Posts")
    return fig

# Callback for greenwashing risk visualization
@app.callback(
    Output('greenwashing-viz', 'figure'),
    [Input('company-filter', 'value'),
     Input('platform-filter', 'value'),
     Input('date-range', 'start_date'),
     Input('date-range', 'end_date')]
)
def update_greenwashing_risk(company, platform, start_date, end_date):
    filtered_df = filter_data(company, platform, start_date, end_date)
    
    if filtered_df.empty:
        fig = go.Figure()
        fig.update_layout(
            title='Greenwashing Risk Assessment',
            xaxis={'visible': False},
            yaxis={'visible': False},
            annotations=[{'text': 'No data available for selected filters', 'showarrow': False, 'font': {'size': 20}}]
        )
        return fig
    
    # List to keep track of posts with valid risk scores
    valid_posts = []
    
    # Calculate risk scores for all posts
    platforms = []
    sentiment_scores = []
    contradiction_scores = []
    risk_scores = []
    sizes = []
    post_ids = []
    dates = []
    texts = []
    
    for idx, row in filtered_df.iterrows():
        platform_std = row['platform_standardized']
        sentiment = row['sentiment_score']
        contradiction = row.get('contradiction_score', 0)
        
        # Calculate comment-based risk
        comment_risk = 0
        comments = row.get('comments', [])
        comment_sentiments = row.get('comment_sentiments', [])
        
        if isinstance(comments, list) and len(comments) > 0 and isinstance(comment_sentiments, list) and len(comment_sentiments) > 0:
            negative_comments = sum(1 for s in comment_sentiments if s == 'negative')
            skeptical_comments = sum(1 for s in comment_sentiments if s == 'skeptical')
            total_comments = len(comment_sentiments)
            
            if total_comments > 0:
                comment_risk = (negative_comments + 0.5 * skeptical_comments) / total_comments
        
        # Calculate overall risk score
        if pd.isna(sentiment) or pd.isna(contradiction) or pd.isna(comment_risk):
            continue
        
        risk_score = (abs(sentiment) + contradiction + comment_risk) / 3
        
        # Only include valid posts
        valid_posts.append(idx)
        platforms.append(platform_std)
        sentiment_scores.append(sentiment)
        contradiction_scores.append(contradiction)
        risk_scores.append(risk_score)
        post_ids.append(row['post_id'])
        dates.append(row['timestamp'])
        texts.append(row['content'][:100] + "..." if len(row['content']) > 100 else row['content'])
    
    # Check if we have any valid data
    if not valid_posts:
        fig = go.Figure()
        fig.update_layout(
            title='Greenwashing Risk Assessment',
            xaxis={'visible': False},
            yaxis={'visible': False},
            annotations=[{'text': 'No valid risk data available', 'showarrow': False, 'font': {'size': 20}}]
        )
        return fig
    
    # Normalize sizes for the scatter plot (between 10 and 40)
    if len(set(risk_scores)) == 1:  # All risk scores are the same
        sizes = [25] * len(risk_scores)
    else:
        min_risk = min(risk_scores)
        max_risk = max(risk_scores)
        sizes = [10 + ((r - min_risk) / (max_risk - min_risk)) * 30 for r in risk_scores]
    
    # Create figure
    fig = go.Figure()
    
    # Create a DataFrame for the plot
    plot_df = pd.DataFrame({
        'platform': platforms,
        'sentiment': sentiment_scores,
        'contradiction': contradiction_scores,
        'risk': risk_scores,
        'size': sizes,
        'post_id': post_ids,
        'date': dates,
        'text': texts
    })
    
    # Ensure no NaN values
    plot_df = plot_df.dropna(subset=['sentiment', 'contradiction', 'risk', 'size'])
    
    # Define platform-specific colors
    platform_colors = {
        'LinkedIn': '#0077B5',
        'Twitter': '#1DA1F2',
        'Facebook': '#4267B2',
        'Instagram': '#C13584',
        'Website': '#6c757d',
        'Report': '#28a745',
        'News': '#fd7e14',
        'Other': '#6f42c1',
        'Unknown': '#495057'
    }
    
    # Add scatter plot with platform-based colors
    for platform_name in plot_df['platform'].unique():
        platform_df = plot_df[plot_df['platform'] == platform_name]
        
        # Get color for this platform (default to gray if not in the map)
        color = platform_colors.get(platform_name, '#6c757d')
        
        fig.add_trace(go.Scatter(
            x=platform_df['sentiment'],
            y=platform_df['contradiction'],
            mode='markers',
            marker=dict(
                size=platform_df['size'],
                color=color,
                opacity=0.7,
                line=dict(width=1, color='white')
            ),
            name=platform_name,
            text=[f"Platform: {p}<br>Date: {d}<br>Risk: {r:.2f}<br>{t}" 
                  for p, d, r, t in zip(platform_df['platform'], platform_df['date'], 
                                     platform_df['risk'], platform_df['text'])],
            hoverinfo='text'
        ))
    
    # Add color scale for risk score
    risk_min = min(plot_df['risk'])
    risk_mean = plot_df['risk'].mean()
    risk_max = max(plot_df['risk'])
    
    # Add color scale legend
    for i, risk_level in enumerate([risk_min, risk_mean, risk_max]):
        y_pos = 1.02 - (i * 0.05)
        fig.add_annotation(
            x=1.02,
            y=y_pos,
            xref="paper",
            yref="paper",
            text=f"Risk: {risk_level:.2f}",
            showarrow=False,
            font=dict(
                size=10,
                color="black"
            ),
            align="left"
        )
    
    # Update layout
    fig.update_layout(
        title='Greenwashing Risk Assessment',
        xaxis=dict(
            title='Sentiment Score',
            showgrid=True,
            zeroline=True,
            range=[-1.1, 1.1]
        ),
        yaxis=dict(
            title='Contradiction Score',
            showgrid=True,
            zeroline=True,
            range=[0, 1.1]
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        margin=dict(l=20, r=20, t=40, b=20),
        hovermode='closest'
    )
    
    # Add explanatory annotations for quadrants
    annotations = [
        dict(x=0.75, y=0.9, text="High Risk:<br>Positive Claims<br>High Contradiction", showarrow=False, 
             font=dict(size=10, color="red"), xref="x", yref="y", align="center", 
             bordercolor="red", borderwidth=2, borderpad=4, bgcolor="white", opacity=0.8),
        
        dict(x=-0.75, y=0.9, text="High Risk:<br>Negative Claims<br>High Contradiction", showarrow=False, 
             font=dict(size=10, color="red"), xref="x", yref="y", align="center", 
             bordercolor="red", borderwidth=2, borderpad=4, bgcolor="white", opacity=0.8),
        
        dict(x=0.75, y=0.1, text="Low Risk:<br>Positive Claims<br>Low Contradiction", showarrow=False, 
             font=dict(size=10, color="green"), xref="x", yref="y", align="center", 
             bordercolor="green", borderwidth=2, borderpad=4, bgcolor="white", opacity=0.8),
        
        dict(x=-0.75, y=0.1, text="Medium Risk:<br>Negative Claims<br>Low Contradiction", showarrow=False, 
             font=dict(size=10, color="orange"), xref="x", yref="y", align="center", 
             bordercolor="orange", borderwidth=2, borderpad=4, bgcolor="white", opacity=0.8)
    ]
    
    fig.update_layout(annotations=annotations)
    
    # Add quadrant lines
    fig.add_shape(type="line", x0=0, y0=0, x1=0, y1=1, line=dict(color="gray", width=1, dash="dash"))
    fig.add_shape(type="line", x0=-1, y0=0.5, x1=1, y1=0.5, line=dict(color="gray", width=1, dash="dash"))
    
    return fig

# Callback for comment sentiment distribution
@app.callback(
    Output('comment-sentiment-distribution', 'figure'),
    [Input('company-filter', 'value'),
     Input('platform-filter', 'value'),
     Input('date-range', 'start_date'),
     Input('date-range', 'end_date')]
)
def update_comment_sentiment_distribution(company, platform, start_date, end_date):
    filtered_df = filter_data(company, platform, start_date, end_date)
    
    if filtered_df.empty:
        return go.Figure(layout={
            'title': 'Comment Sentiment Distribution',
            'xaxis': {'visible': False}, 'yaxis': {'visible': False},
            'annotations': [{'text': 'No data for selected filters', 'showarrow': False}]
        })
        
    all_comment_sentiments = []
    for sentiments in filtered_df['comment_sentiments']:
        if isinstance(sentiments, list):
            all_comment_sentiments.extend(sentiments)
            
    if not all_comment_sentiments:
         return go.Figure(layout={
            'title': 'Comment Sentiment Distribution',
            'xaxis': {'visible': False}, 'yaxis': {'visible': False},
            'annotations': [{'text': 'No comments found', 'showarrow': False}]
        })

    sentiment_counts = pd.Series(all_comment_sentiments).value_counts().reset_index()
    sentiment_counts.columns = ['Sentiment', 'Count']
    # Ensure consistent categories and order
    sentiment_map = {'positive': 'Positive', 'negative': 'Negative', 'skeptical': 'Skeptical'}
    sentiment_counts['Sentiment'] = sentiment_counts['Sentiment'].map(sentiment_map).fillna('Unknown')
    sentiment_order = ['Positive', 'Skeptical', 'Negative', 'Unknown']
    sentiment_counts = sentiment_counts.set_index('Sentiment').reindex(sentiment_order).fillna(0).reset_index()
    
    fig = px.pie(
        sentiment_counts,
        values='Count',
        names='Sentiment',
        title='Comment Sentiment Distribution',
        color='Sentiment',
        color_discrete_map={'Positive': 'green', 'Negative': 'red', 'Skeptical': 'orange', 'Unknown':'grey'}
        # template='plotly_white' # Set globally now
    )
    fig.update_traces(textposition='inside', textinfo='percent+label')
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
    filtered_df = filter_data(company, platform, start_date, end_date)
    
    if filtered_df.empty:
        return go.Figure(layout={
            'title': 'Comment Volume Over Time',
            'xaxis': {'visible': False}, 'yaxis': {'visible': False},
            'annotations': [{'text': 'No data for selected filters', 'showarrow': False}]
        })
        
    # Calculate comment count per post
    filtered_df['comment_count'] = filtered_df['comments'].apply(lambda x: len(x) if isinstance(x, list) else 0)
    
    try:
        # Convert timestamp to datetime and handle potential errors
        filtered_df['timestamp'] = pd.to_datetime(filtered_df['timestamp'], errors='coerce')
        
        # Drop any rows where timestamp conversion failed
        filtered_df = filtered_df.dropna(subset=['timestamp'])
        
        if filtered_df.empty:
            return go.Figure(layout={
                'title': 'Comment Volume Over Time',
                'xaxis': {'visible': False}, 'yaxis': {'visible': False},
                'annotations': [{'text': 'No valid timestamps in data', 'showarrow': False}]
            })
        
        # Sort by timestamp
        filtered_df = filtered_df.sort_values('timestamp')
        
        # Create a copy with timestamp as index for resampling
        df_resampled = filtered_df.set_index('timestamp')
        
        # Resample by day and calculate rolling average
        daily_comments = df_resampled['comment_count'].resample('D').sum().reset_index()
        daily_comments['rolling_avg'] = daily_comments['comment_count'].rolling(window=7, min_periods=1, center=True).mean()
        
    except Exception as e:
        print(f"Error in resampling: {str(e)}")
        # Fallback: aggregate manually by date
        filtered_df['date'] = filtered_df['timestamp'].dt.date
        daily_comments = filtered_df.groupby('date', as_index=False)['comment_count'].sum()
        daily_comments.columns = ['timestamp', 'comment_count']
        daily_comments['rolling_avg'] = daily_comments['comment_count'].rolling(window=7, min_periods=1, center=True).mean()

    if daily_comments.empty:
        return go.Figure(layout={
            'title': 'Comment Volume Over Time',
            'xaxis': {'visible': False}, 'yaxis': {'visible': False},
            'annotations': [{'text': 'No comments to plot', 'showarrow': False}]
        })
    
    # Create the figure
    fig = px.line(
        daily_comments,
        x='timestamp',
        y=['comment_count', 'rolling_avg'],
        title='Comment Volume Over Time',
        labels={'timestamp': 'Date', 'value': 'Number of Comments'}
    )
    
    # Update trace names and layout
    fig.data[0].name = 'Daily Count'
    fig.data[1].name = 'Rolling Avg (7 days)'
    fig.update_layout(
        legend_title_text='Metric',
        hovermode='x unified'
    )
    fig.update_traces(hovertemplate='%{y:.0f} comments')
    
    return fig

# Callback for comment summary
@app.callback(
    Output('comment-summary', 'children'),
    [Input('company-filter', 'value'),
     Input('platform-filter', 'value'),
     Input('date-range', 'start_date'),
     Input('date-range', 'end_date')]
)
def update_comment_summary(company, platform, start_date, end_date):
    filtered_df = filter_data(company, platform, start_date, end_date)
    
    if filtered_df.empty:
        return html.P("No comment data available for selected filters.")
        
    total_posts = len(filtered_df)
    total_comments = sum(len(c) if isinstance(c, list) else 0 for c in filtered_df['comments'])
    avg_comments_per_post = total_comments / total_posts if total_posts > 0 else 0
    
    all_comment_sentiments = []
    for sentiments in filtered_df['comment_sentiments']:
        if isinstance(sentiments, list):
            all_comment_sentiments.extend(sentiments)
            
    sentiment_counts = pd.Series(all_comment_sentiments).value_counts()
    positive_comments = sentiment_counts.get('positive', 0)
    negative_comments = sentiment_counts.get('negative', 0)
    skeptical_comments = sentiment_counts.get('skeptical', 0)

    return [
        html.H5("Comment Statistics", className="text-info"),
        dbc.Row([
            dbc.Col(html.Div([html.Strong("Total Comments:"), f" {total_comments}"]), width=6),
            dbc.Col(html.Div([html.Strong("Avg Comments/Post:"), f" {avg_comments_per_post:.1f}"]), width=6),
        ]),
        dbc.Row([
            dbc.Col(html.Div([html.Strong("Positive Comments:"), f" {positive_comments}"]), width=4),
            dbc.Col(html.Div([html.Strong("Skeptical Comments:"), f" {skeptical_comments}"]), width=4),
            dbc.Col(html.Div([html.Strong("Negative Comments:"), f" {negative_comments}"]), width=4),
        ], className="mt-2")
    ]

# Callback for detailed analysis
@app.callback(
    Output('detailed-analysis', 'children'),
    [Input('company-filter', 'value'),
     Input('platform-filter', 'value'),
     Input('date-range', 'start_date'),
     Input('date-range', 'end_date')]
)
def update_detailed_analysis(company, platform, start_date, end_date):
    filtered_df = filter_data(company, platform, start_date, end_date)
    
    if filtered_df.empty:
        return html.P("No data available for detailed analysis.")
        
    # --- Recalculate risk score for the current filtered data --- 
    # Ensure alignment and availability of the risk score column
    risk_scores_detailed = []
    for _, row in filtered_df.iterrows():
        sentiment = row['sentiment_score']
        contradiction = row['contradiction_score']
        comments_list = row['comment_sentiments']
        
        if not isinstance(comments_list, list) or not comments_list:
            comment_risk = 0
        else:
            negative_comments = sum(1 for s in comments_list if s == 'negative')
            skeptical_comments = sum(1 for s in comments_list if s == 'skeptical')
            total_comments = len(comments_list)
            comment_risk = (negative_comments + 0.5 * skeptical_comments) / total_comments if total_comments > 0 else 0
            
        if pd.isna(sentiment) or pd.isna(contradiction) or pd.isna(comment_risk):
             risk_score = np.nan
        else:
            risk_score = (abs(sentiment) + contradiction + comment_risk) / 3
        risk_scores_detailed.append(risk_score)
        
    # Add the calculated risk score column to this specific filtered_df
    filtered_df['risk_score_detailed'] = risk_scores_detailed
    # ------------------------------------------------------------
    
    # Key Metrics
    avg_sentiment = filtered_df['sentiment_score'].mean()
    avg_contradiction = filtered_df['contradiction_score'].mean()
    total_posts = len(filtered_df)
    total_comments = sum(len(c) if isinstance(c, list) else 0 for c in filtered_df['comments'])

    # Top Emotions
    emotions_list = []
    for content in filtered_df['content']:
        if pd.notna(content):
            emotions = emotion_classifier.get_sustainability_emotions(content)
            if emotions.get('has_sustainability_context', False):
                emotions_list.extend(emotions.get('emotions', {}).keys())
    top_emotions = pd.Series(emotions_list).value_counts().head(5).index.tolist()

    # Greenwashing Indicators (using the locally calculated risk score)
    # Filter using the aligned 'risk_score_detailed' column
    high_risk_posts = filtered_df[filtered_df['risk_score_detailed'].notna() & (filtered_df['risk_score_detailed'] > 0.7)] 
    high_contradiction_posts = filtered_df[filtered_df['contradiction_score'] > 0.7]
    num_high_risk = len(high_risk_posts)
    num_high_contradiction = len(high_contradiction_posts)
    risk_percentage = (num_high_risk / total_posts * 100) if total_posts > 0 else 0
    contradiction_percentage = (num_high_contradiction / total_posts * 100) if total_posts > 0 else 0

    # Extract indicators from emotion analysis (placeholder)
    indicators_from_emotion = {'Excessive Optimism': 0, 'Lack of Authenticity': 0, 'Overconfidence': 0}

    return [
        dbc.Row([
            dbc.Col([
                html.H5("Key Metrics", className="text-info"),
                html.Ul([
                    html.Li(f"Total Posts Analyzed: {total_posts}"),
                    html.Li(f"Total Comments Analyzed: {total_comments}"),
                    html.Li(f"Average Post Sentiment: {avg_sentiment:.2f}"),
                    html.Li(f"Average Contradiction Score: {avg_contradiction:.2f}")
                ])
            ], md=6),
            dbc.Col([
                html.H5("Potential Greenwashing Indicators", className="text-danger"),
                html.Ul([
                    html.Li(f"Posts with High Risk Score (> 0.7): {num_high_risk} ({risk_percentage:.1f}% of posts)"),
                    html.Li(f"Posts with High Contradiction Score (> 0.7): {num_high_contradiction} ({contradiction_percentage:.1f}% of posts)"),
                    # Add emotion-based indicators here when available
                    # html.Li(f"Excessive Optimism Flags: {indicators_from_emotion['Excessive Optimism']}"),
                    # html.Li(f"Lack of Authenticity Flags: {indicators_from_emotion['Lack of Authenticity']}"),
                    # html.Li(f"Overconfidence Flags: {indicators_from_emotion['Overconfidence']}")
                ])
            ], md=6)
        ]),
        dbc.Row([
            dbc.Col([
                 html.H5("Top 5 Emotions Detected", className="text-info mt-3"),
                 html.Ul([html.Li(e) for e in top_emotions]) if top_emotions else html.P("No sustainability-related emotions detected.")
            ])
        ], className="mt-3")
    ]

# Callback for company risk score
@app.callback(
    [Output('company-risk-score', 'children'),
     Output('company-risk-description', 'children')],
    [Input('company-filter', 'value')]
)
def update_company_risk(company):
    if not company or df.empty:
        return "N/A", "No data available"
    
    company_df = df[df['company'] == company].copy()
    if company_df.empty:
        return "N/A", "No data for this company"
    
    # Calculate risk scores for all posts
    risk_scores = []
    
    for idx, row in company_df.iterrows():
        sentiment = row.get('sentiment_score', 0)
        contradiction = row.get('contradiction_score', 0)
        
        # Calculate comment-based risk
        comment_risk = 0
        if 'comment_sentiments' in row and isinstance(row['comment_sentiments'], list):
            sentiments = row['comment_sentiments']
            if sentiments:
                negative_comments = sum(1 for s in sentiments if s == 'negative')
                skeptical_comments = sum(1 for s in sentiments if s == 'skeptical')
                total_comments = len(sentiments)
                
                if total_comments > 0:
                    comment_risk = (negative_comments + 0.5 * skeptical_comments) / total_comments
        
        # Calculate overall risk score
        risk_score = (abs(sentiment) + contradiction + comment_risk) / 3
        risk_scores.append(risk_score)
    
    # Calculate average risk score (ignore NaN values)
    valid_risk_scores = [score for score in risk_scores if not pd.isna(score)]
    
    if not valid_risk_scores:
        return "N/A", "Could not calculate risk score"
    
    avg_risk = sum(valid_risk_scores) / len(valid_risk_scores)
    
    # Determine risk level and color
    if avg_risk > 0.7:
        risk_level = "High"
        color = "danger"
    elif avg_risk > 0.4:
        risk_level = "Medium"
        color = "warning"
    else:
        risk_level = "Low"
        color = "success"
    
    return f"{avg_risk:.2f}", html.Span(f"{risk_level} Risk", className=f"badge bg-{color}")

# Callback for top contradictions
@app.callback(
    Output('top-contradictions', 'children'),
    [Input('company-filter', 'value'),
     Input('platform-filter', 'value'),
     Input('date-range', 'start_date'),
     Input('date-range', 'end_date')]
)
def update_top_contradictions(company, platform, start_date, end_date):
    filtered_df = filter_data(company, platform, start_date, end_date)
    
    if filtered_df.empty:
        return html.Div([
            html.H5("Top Contradiction Posts"),
            html.P("No data available for selected filters", className="text-muted")
        ])
    
    # Find posts with highest contradiction scores
    filtered_df = filtered_df.sort_values('contradiction_score', ascending=False)
    top_posts = filtered_df.head(3)
    
    if top_posts.empty or top_posts['contradiction_score'].isna().all():
        return html.Div([
            html.H5("Top Contradiction Posts"),
            html.P("No contradiction data available", className="text-muted")
        ])
    
    # Create cards for top contradiction posts
    cards = []
    for idx, row in top_posts.iterrows():
        # Create sentiment badge
        sentiment = row.get('sentiment_score', 0)
        if sentiment > 0.3:
            sentiment_badge = html.Span("Positive", className="badge bg-success me-2")
        elif sentiment > -0.3:
            sentiment_badge = html.Span("Neutral", className="badge bg-secondary me-2")
        else:
            sentiment_badge = html.Span("Negative", className="badge bg-danger me-2")
        
        # Create platform badge
        platform_name = row.get('platform_standardized', 'Unknown')
        platform_badge = html.Span(platform_name, className="badge bg-info me-2")
        
        # Format date
        post_date = row.get('timestamp', '')
        date_str = post_date.strftime('%Y-%m-%d') if isinstance(post_date, (datetime, pd.Timestamp)) else 'Unknown Date'
        
        # Get comment count
        comments = row.get('comments', [])
        comment_count = len(comments) if isinstance(comments, list) else 0
        
        cards.append(
            dbc.Card([
                dbc.CardBody([
                    html.Div([
                        platform_badge,
                        sentiment_badge,
                        html.Span(f"Contradiction: {row.get('contradiction_score', 0):.2f}", className="badge bg-warning")
                    ], className="mb-2"),
                    html.P(row.get('content', '')[:200] + "..." if len(row.get('content', '')) > 200 else row.get('content', ''), 
                           className="mb-2"),
                    html.Div([
                        html.Small(f"{date_str} • {comment_count} comments", className="text-muted")
                    ], className="d-flex justify-content-between")
                ])
            ], className="mb-3")
        )
    
    return html.Div([
        html.H5("Top Contradiction Posts"),
        html.Div(cards)
    ])

# Callback for posts list
@app.callback(
    Output('posts-list', 'children'),
    [Input('company-filter', 'value'),
     Input('platform-filter', 'value'),
     Input('date-range', 'start_date'),
     Input('date-range', 'end_date')]
)
def update_posts_list(company, platform, start_date, end_date):
    filtered_df = filter_data(company, platform, start_date, end_date)
    
    if filtered_df.empty:
        return html.Div([
            html.P("No posts found with the current filters", className="text-muted")
        ])
    
    # Sort by date, most recent first
    filtered_df = filtered_df.sort_values('timestamp', ascending=False)
    
    # Take top 10 posts
    display_posts = filtered_df.head(10)
    
    # Create post list items
    post_items = []
    for idx, row in display_posts.iterrows():
        # Create sentiment badge based on sentiment score
        sentiment = row.get('sentiment_score', 0)
        if sentiment > 0.3:
            sentiment_badge = html.Span("Positive", className="badge bg-success me-2")
            sentiment_color = "success"
        elif sentiment > -0.3:
            sentiment_badge = html.Span("Neutral", className="badge bg-secondary me-2")
            sentiment_color = "secondary"
        else:
            sentiment_badge = html.Span("Negative", className="badge bg-danger me-2")
            sentiment_color = "danger"
            
        # Determine risk level
        contradiction = row.get('contradiction_score', 0)
        comments = row.get('comments', [])
        comment_sentiments = row.get('comment_sentiments', [])
        
        # Calculate comment-based risk
        comment_risk = 0
        if isinstance(comments, list) and len(comments) > 0 and isinstance(comment_sentiments, list) and len(comment_sentiments) > 0:
            negative_comments = sum(1 for s in comment_sentiments if s == 'negative')
            skeptical_comments = sum(1 for s in comment_sentiments if s == 'skeptical')
            total_comments = len(comment_sentiments)
            
            if total_comments > 0:
                comment_risk = (negative_comments + 0.5 * skeptical_comments) / total_comments
        
        # Calculate overall risk
        risk_score = (abs(sentiment) + contradiction + comment_risk) / 3
        
        if risk_score > 0.7:
            risk_badge = html.Span("High Risk", className="badge bg-danger me-2")
        elif risk_score > 0.4:
            risk_badge = html.Span("Medium Risk", className="badge bg-warning me-2")
        else:
            risk_badge = html.Span("Low Risk", className="badge bg-success me-2")
        
        # Format date
        post_date = row.get('timestamp', '')
        date_str = post_date.strftime('%Y-%m-%d') if isinstance(post_date, (datetime, pd.Timestamp)) else 'Unknown Date'
        
        # Get platform
        platform_name = row.get('platform_standardized', 'Unknown')
        platform_badge = html.Span(platform_name, className="badge bg-info me-2")
        
        # Get comment count
        comment_count = len(comments) if isinstance(comments, list) else 0
        
        # Create engagement section if available
        engagement_section = html.Div(className="mt-2")
        engagement_metrics = row.get('engagement_metrics', {})
        if isinstance(engagement_metrics, dict) and engagement_metrics:
            engagement_items = []
            for metric, value in engagement_metrics.items():
                if isinstance(value, (int, float)) and not pd.isna(value):
                    engagement_items.append(
                        html.Span(f"{metric.capitalize()}: {value}", className="me-3 text-muted small")
                    )
            
            if engagement_items:
                engagement_section = html.Div(engagement_items, className="mt-2 d-flex flex-wrap")
        
        # Create post card
        post_card = dbc.Card([
            dbc.CardBody([
                html.Div([
                    platform_badge,
                    sentiment_badge,
                    risk_badge,
                    html.Span(f"ID: {row.get('post_id', 'N/A')}", className="small text-muted ms-2")
                ], className="mb-2"),
                
                html.P(row.get('content', 'No content'), className="mb-2"),
                
                engagement_section,
                
                # Comments preview if available
                html.Div([
                    html.Div(f"Comments ({comment_count}):", className="mt-2 mb-1 small fw-bold") if comment_count > 0 else None,
                    html.Div([
                        html.P(
                            comments[i][:100] + "..." if len(comments[i]) > 100 else comments[i],
                            className=f"small mb-1 text-{sentiment_color if comment_sentiments[i] == 'positive' else 'warning' if comment_sentiments[i] == 'skeptical' else 'danger'}"
                        ) for i in range(min(3, comment_count))
                    ]) if comment_count > 0 else None,
                    html.P(f"+ {comment_count - 3} more comments", className="small text-muted") if comment_count > 3 else None
                ]),
                
                html.Div([
                    html.Small(f"{date_str}", className="text-muted")
                ], className="d-flex justify-content-between mt-2")
            ])
        ], className="mb-3")
        
        post_items.append(post_card)
    
    # Create pagination if more than 10 posts
    pagination = None
    if len(filtered_df) > 10:
        pagination = html.Div([
            html.P(f"Showing 10 of {len(filtered_df)} posts", className="text-muted text-center"),
            dbc.Pagination(
                id="posts-pagination",
                max_value=int(np.ceil(len(filtered_df) / 10)),
                first_last=True,
                previous_next=True,
                active_page=1,
                className="justify-content-center"
            )
        ], className="mt-3")
    
    return html.Div([
        html.Div(post_items),
        pagination
    ])

# Callback for data format description
@app.callback(
    Output('format-description', 'children'),
    Input('data-format-dropdown', 'value')
)
def update_format_description(selected_format):
    if selected_format == 'standard':
        return html.Div([
            html.H6("Standard Format", className="text-primary"),
            html.P("The standard format includes all required fields with JSON-formatted data for engagement metrics and comments."),
            html.Hr(),
            html.H6("Required Columns:", className="mt-3"),
            html.Ul([
                html.Li("post_id: Unique identifier for each post"),
                html.Li("company: Company name"),
                html.Li("platform: Social media platform"),
                html.Li("timestamp: Date and time of the post"),
                html.Li("content: Text content of the post"),
                html.Li("engagement_metrics: JSON string with metrics"),
                html.Li("comments: JSON array of comment strings")
            ]),
            html.Hr(),
            html.H6("Example:", className="mt-3"),
            html.Pre("""
post_id,company,platform,timestamp,content,engagement_metrics,comments
1,GreenTech,Twitter,2024-03-15 10:30:00,"New sustainable packaging! 🌱",{"likes": 150, "retweets": 45},["Great!", "Data?", "Promising"]
            """, className="bg-light p-2 rounded small"),
            html.H5("Example Data", className="mt-4 text-primary"),
            html.Pre("""
post_id,company,platform,timestamp,content,engagement_metrics,comments
1,GreenTech,Twitter,2024-03-15 10:30:00,"New sustainable packaging! 🌱",{"likes": 150, "retweets": 45},["Great!", "Data?", "Promising"]
2,EcoCorp,LinkedIn,2024-03-16 09:15:00,"Carbon neutrality by 2025.",{"likes": 450, "shares": 85},["Bold", "How?", "Updates?"]
""", className="bg-white p-3 rounded small border"),
            html.H5("Notes", className="mt-4 text-primary"),
            html.Ul([
                html.Li("Ensure `engagement_metrics` and `comments` are valid JSON.", className="mb-2"),
                html.Li("Timestamps need the specified format for time analysis.", className="mb-2"),
                html.Li("Sentiment and contradiction scores are calculated automatically.", className="mb-2")
            ])
        ])
    elif selected_format == 'simple':
        return html.Div([
            html.H6("Simple Format (No JSON)", className="text-primary"),
            html.P("A simplified format that doesn't require JSON parsing for engagement metrics and comments."),
            html.Hr(),
            html.H6("Required Columns:", className="mt-3"),
            html.Ul([
                html.Li("post_id: Unique identifier for each post"),
                html.Li("company: Company name"),
                html.Li("platform: Social media platform"),
                html.Li("timestamp: Date and time of the post"),
                html.Li("content: Text content of the post"),
                html.Li("likes: Number of likes (integer)"),
                html.Li("shares: Number of shares (integer)"),
                html.Li("comments: Pipe-separated list of comments")
            ]),
            html.Hr(),
            html.H6("Example:", className="mt-3"),
            html.Pre("""
post_id,company,platform,timestamp,content,likes,shares,comments
1,GreenTech,Twitter,2024-03-15 10:30:00,"New sustainable packaging! 🌱",150,45,"Great!|Data?|Promising"
            """, className="bg-light p-2 rounded small")
        ])
    elif selected_format == 'extended':
        return html.Div([
            html.H6("Extended Format (With Sentiment)", className="text-primary"),
            html.P("An extended format that includes pre-calculated sentiment scores and additional metadata."),
            html.Hr(),
            html.H6("Required Columns:", className="mt-3"),
            html.Ul([
                html.Li("post_id: Unique identifier for each post"),
                html.Li("company: Company name"),
                html.Li("platform: Social media platform"),
                html.Li("timestamp: Date and time of the post"),
                html.Li("content: Text content of the post"),
                html.Li("engagement_metrics: JSON string with metrics"),
                html.Li("comments: JSON array of comment strings"),
                html.Li("sentiment_score: Pre-calculated sentiment score (-1 to 1)"),
                html.Li("contradiction_score: Pre-calculated contradiction score (0 to 1)"),
                html.Li("comment_sentiments: JSON array of comment sentiment labels")
            ]),
            html.Hr(),
            html.H6("Example:", className="mt-3"),
            html.Pre("""
post_id,company,platform,timestamp,content,engagement_metrics,comments,sentiment_score,contradiction_score,comment_sentiments
1,GreenTech,Twitter,2024-03-15 10:30:00,"New sustainable packaging! 🌱",{"likes": 150, "retweets": 45},["Great!", "Data?", "Promising"],0.8,0.2,["positive", "skeptical", "positive"]
            """, className="bg-light p-2 rounded small")
        ])
    else:  # minimal
        return html.Div([
            html.H6("Minimal Format", className="text-primary"),
            html.P("A minimal format with only the essential fields needed for basic analysis."),
            html.Hr(),
            html.H6("Required Columns:", className="mt-3"),
            html.Ul([
                html.Li("company: Company name"),
                html.Li("platform: Social media platform"),
                html.Li("timestamp: Date and time of the post"),
                html.Li("content: Text content of the post")
            ]),
            html.Hr(),
            html.H6("Example:", className="mt-3"),
            html.Pre("""
company,platform,timestamp,content
GreenTech,Twitter,2024-03-15 10:30:00,"New sustainable packaging! 🌱"
            """, className="bg-light p-2 rounded small")
        ])

# Run the app
if __name__ == '__main__':
    app.run(debug=True) 