import dash
from dash import html, dcc, Input, Output, State
import dash_bootstrap_components as dbc
from dash_bootstrap_components import NavbarSimple, NavItem, NavLink
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sentiment_analyzer import SentimentAnalyzer
from emotion_classifier import EmotionClassifier
from contradiction_detector import ContradictionDetector
from data_generator import DataGenerator
import base64
import io
import json

# Set default plotly template
pio.templates.default = "plotly_white"

# Initialize the Dash app with a Bootstrap theme
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])

# Initialize analyzers
sentiment_analyzer = SentimentAnalyzer()
emotion_classifier = EmotionClassifier()
contradiction_detector = ContradictionDetector()
data_generator = DataGenerator()

# Generate initial synthetic data
df = data_generator.generate_synthetic_data(100)
data_generator.save_to_csv(df)

# Define the layout
app.layout = dbc.Container(fluid=True, children=[
    # Header Navbar
    NavbarSimple(
        children=[
            NavItem(NavLink("GitHub", href="https://github.com/bumincetin/greenwashing-detection", target="_blank"))
        ],
        brand="Greenwashing Detection Dashboard",
        brand_href="#",
        color="primary",
        dark=True,
        className="mb-4"
    ),

    # Main content area
    dbc.Container([
        # Data Format Section
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Data Format and Structure"),
                    dbc.CardBody([
                        html.H5("Required Data Format", className="card-title text-info"),
                        html.P("Upload a CSV file with the following columns:"),
                        html.Ul([
                            html.Li([html.Code("post_id"), ": Unique identifier (integer)"]),
                            html.Li([html.Code("company"), ": Company name (string)"]),
                            html.Li([html.Code("platform"), ": Social media platform (string)"]),
                            html.Li([html.Code("timestamp"), ": Timestamp (YYYY-MM-DD HH:MM:SS)"]),
                            html.Li([html.Code("content"), ": Post text (string)"]),
                            html.Li([html.Code("engagement_metrics"), ": JSON string of metrics"]),
                            html.Li([html.Code("comments"), ": JSON array of comment strings"])
                        ], className="list-unstyled"),
                        html.H5("Example Data", className="mt-4 text-info"),
                        html.Pre("""
post_id,company,platform,timestamp,content,engagement_metrics,comments
1,GreenTech,Twitter,2024-03-15 10:30:00,"New sustainable packaging! 🌱",{"likes": 150, "retweets": 45},["Great!", "Data?", "Promising"]
2,EcoCorp,LinkedIn,2024-03-16 09:15:00,"Carbon neutrality by 2025.",{"likes": 450, "shares": 85},["Bold", "How?", "Updates?"]
                        """, className="bg-light p-3 rounded small"),
                        html.H5("Notes", className="mt-4 text-info"),
                        html.Ul([
                            html.Li("Ensure `engagement_metrics` and `comments` are valid JSON."),
                            html.Li("Timestamps need the specified format for time analysis."),
                            html.Li("Sentiment and contradiction scores are calculated automatically.")
                        ])
                    ])
                ], className="mb-4 shadow-sm")
            ])
        ]),

        # File Upload Section
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Upload Your Data"),
                    dbc.CardBody([
                        dcc.Upload(
                            id='upload-data',
                            children=html.Div([
                                'Drag and Drop or ', html.A('Select CSV File')
                            ]),
                            style={
                                'width': '100%', 'height': '60px', 'lineHeight': '60px',
                                'borderWidth': '1px', 'borderStyle': 'dashed',
                                'borderRadius': '5px', 'textAlign': 'center', 'margin': '10px 0'
                            },
                            multiple=False
                        ),
                        html.Div(id='upload-status', className="mt-2 text-muted")
                    ])
                ], className="mb-4 shadow-sm")
            ])
        ]),

        # Filters & Company Overview Section
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Filters & Company Overview"),
                    dbc.CardBody([
                        dbc.Row([
                            # Filters Column
                            dbc.Col([
                                html.Label("Company", className="fw-bold"),
                                dcc.Dropdown(
                                    id='company-filter',
                                    options=[{'label': company, 'value': company} for company in df['company'].unique()],
                                    value=df['company'].unique()[0],
                                    clearable=False
                                ),
                                html.Label("Platform", className="mt-3 fw-bold"),
                                dcc.Dropdown(
                                    id='platform-filter',
                                    options=[{'label': platform, 'value': platform} for platform in df['platform'].unique()],
                                    value=df['platform'].unique()[0],
                                    clearable=False
                                ),
                                html.Label("Date Range", className="mt-3 fw-bold"),
                                dcc.DatePickerRange(
                                    id='date-range',
                                    start_date=df['timestamp'].min(),
                                    end_date=df['timestamp'].max(),
                                    display_format='YYYY-MM-DD',
                                    className="d-block"
                                )
                            ], md=6),
                            # Company Risk Score Column
                            dbc.Col([
                                html.Div([
                                    html.H5("Company Greenwashing Risk", className="text-primary"),
                                    html.Div(id='company-risk-score', className="display-4 fw-bold"),
                                    html.Div(id='company-risk-description', className="text-muted small mt-2")
                                ], className="mt-3 p-3 bg-light rounded text-center")
                            ], md=6, className="d-flex align-items-center justify-content-center")
                        ])
                    ])
                ], className="mb-4 shadow-sm")
            ])
        ]),

        # Main Analysis Tabs
        dbc.Tabs([
            dbc.Tab(label="Post Analysis", tab_id="tab-post", children=[
                dbc.Card(dbc.CardBody([                    
                    dbc.Row([
                        dbc.Col(dcc.Graph(id='sentiment-trend'), md=12),
                    ], className="mb-3"),
                    dbc.Row([
                        dbc.Col(dcc.Graph(id='emotion-distribution'), md=6),
                        dbc.Col(dcc.Graph(id='greenwashing-risk'), md=6),
                    ])
                ]), className="mt-3")
            ]),
            dbc.Tab(label="Comment Analysis", tab_id="tab-comment", children=[
                dbc.Card(dbc.CardBody([
                    dbc.Row([
                        dbc.Col(dcc.Graph(id='comment-sentiment-distribution'), md=6),
                        dbc.Col(dcc.Graph(id='comment-engagement-trend'), md=6),
                    ]),
                    dbc.Row([
                         dbc.Col(html.Div(id='comment-summary', className="mt-3 p-3 bg-light rounded"))
                    ])
                ]), className="mt-3")
            ]),
            dbc.Tab(label="Detailed Analysis", tab_id="tab-detailed", children=[
                 dbc.Card(dbc.CardBody(html.Div(id='detailed-analysis')), className="mt-3")
            ]),
        ], id="analysis-tabs", active_tab="tab-post", className="mb-4"),


        # Methodology Section (Accordion)
        dbc.Row([
            dbc.Col([
                dbc.Accordion([
                    dbc.AccordionItem([
                        html.H5("1. Data Format and Upload", className="text-info"),
                        html.P("Start by uploading your social media data in the specified CSV format. The system requires columns like post content, timestamps, engagement metrics (JSON), and comments (JSON array)."),
                        html.H5("2. Filtering Data", className="text-info mt-3"),
                        html.P("Use the dropdowns to select the company and platform you want to analyze. You can also specify a date range to focus on a specific period."),
                        html.H5("3. Analysis Tabs", className="text-info mt-3"),
                        html.Ul([
                            html.Li([html.Strong("Post Analysis:"), " Shows overall sentiment trends over time, the distribution of detected emotions in posts, and a scatter plot visualizing potential greenwashing risk based on post sentiment vs. calculated contradiction score."]),
                            html.Li([html.Strong("Comment Analysis:"), " Displays the sentiment breakdown of user comments (Positive, Skeptical, Negative), tracks comment volume over time, and provides summary statistics."]),
                            html.Li([html.Strong("Detailed Analysis:"), " Provides key numerical metrics like average sentiment, average contradiction score, total posts/comments, top emotions, and flags potential greenwashing indicators."])
                        ]),
                        html.H5("4. Interpretation Notes", className="text-info mt-3"),
                        html.P("Refer to the specific methodology sections below for details on how each metric (Sentiment, Emotion, Contradiction, Risk) is calculated and how to interpret the visualizations.")
                        
                    ], title="Dashboard Usage Guide"),
                    dbc.AccordionItem([
                        html.H5("Sentiment Analysis (Ensemble)", className="text-info"),
                        html.Ul([
                            html.Li("Combines VADER (30%), TextBlob (20%), RoBERTa (30%), DistilBERT (20%)."),
                            html.Li("Scores normalized to [-1 (Negative) to +1 (Positive)]."),
                            html.Li("Trend line uses a 3-point rolling average."),
                            html.Li("Interpretation: Tracks overall tone shifts in communication.")
                        ])
                    ], title="Sentiment Calculation"),
                    dbc.AccordionItem([
                         html.H5("Emotion Classification", className="text-info"),
                         html.Ul([
                            html.Li("Uses GoEmotions model, checking for sustainability keywords."),
                            html.Li("Indicators: High optimism (>0.8), joy (>0.9), pride (>0.8) might suggest greenwashing."),
                            html.Li("Interpretation: Analyzes the emotional undercurrent of posts.")
                         ])
                    ], title="Emotion Analysis"),
                     dbc.AccordionItem([
                         html.H5("Contradiction & Risk Score", className="text-info"),
                         html.Ul([
                             html.Li("Contradiction score (Scatter Y-axis): Calculated based on comment sentiment disagreement with post claims (negative comments weighted higher than skeptical). Scale: 0-1."),
                             html.Li(["Risk Score (Scatter point size/color & Company Score): ", html.Code("(|post_sentiment| + contradiction + comment_risk) / 3"), ". Where ", html.Code("comment_risk = (negative + 0.5 * skeptical) / total_comments"),". Scale: 0-1."]),
                             html.Li("Interpretation: High scores suggest potential greenwashing (discrepancy between claims and feedback). Points in the top-right of the scatter plot are highest risk.")
                         ]),
                     ], title="Contradiction & Risk Calculation"),
                ], start_collapsed=True, flush=True, className="shadow-sm")
            ])
        ], className="mb-5")
    ])
])

# Callback for methodology collapse (Removed as Accordion handles this)

# Callback for data upload
@app.callback(
    [Output('upload-status', 'children'),
     Output('company-filter', 'options'),
     Output('company-filter', 'value'), # Reset value on new upload
     Output('platform-filter', 'options'),
     Output('platform-filter', 'value'), # Reset value on new upload
     Output('date-range', 'start_date'), # Reset date range
     Output('date-range', 'end_date')],  # Reset date range
    Input('upload-data', 'contents'),
    State('upload-data', 'filename')
)
def update_output(contents, filename):
    global df # Allow modification of the global DataFrame
    if contents is None:
        # Keep existing data if no file is uploaded initially
        first_company = df['company'].unique()[0] if not df.empty else None
        first_platform = df['platform'].unique()[0] if not df.empty else None
        min_date = df['timestamp'].min() if not df.empty else None
        max_date = df['timestamp'].max() if not df.empty else None
        company_options = [{'label': c, 'value': c} for c in df['company'].unique()] if not df.empty else []
        platform_options = [{'label': p, 'value': p} for p in df['platform'].unique()] if not df.empty else []
        
        return (dash.no_update, company_options, first_company, platform_options, first_platform, min_date, max_date)
    
    status_message = dash.no_update # Default status
    company_options, first_company = [], None
    platform_options, first_platform = [], None
    min_date, max_date = None, None
    
    try:
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        # Assume CSV for now
        if 'csv' in filename:
            # Read the uploaded CSV data
            new_df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
            
            # --- Data Processing & Validation --- 
            required_columns = ['post_id', 'company', 'platform', 'timestamp', 'content', 'engagement_metrics', 'comments']
            if not all(col in new_df.columns for col in required_columns):
                 raise ValueError(f"Missing required columns. Ensure file has: {', '.join(required_columns)}")
                 
            # Convert timestamp
            try:
                new_df['timestamp'] = pd.to_datetime(new_df['timestamp'])
            except Exception as e:
                raise ValueError(f"Timestamp format error. Use YYYY-MM-DD HH:MM:SS. Details: {e}")

            # Process JSON columns with error handling
            for col in ['engagement_metrics', 'comments']:
                try:
                    # Use json.loads for parsing JSON strings
                    new_df[col] = new_df[col].apply(lambda x: json.loads(x) if pd.notna(x) and isinstance(x, str) else ({} if col == 'engagement_metrics' else []))
                except json.JSONDecodeError as e:
                    # Find the row number causing the error
                    problematic_row = new_df[new_df[col].apply(lambda x: isinstance(x, str) and not x.startswith(('[', '{')))].index.tolist()
                    row_info = f" near row {problematic_row[0]+1}" if problematic_row else ""
                    raise ValueError(f"Error decoding JSON in column '{col}'{row_info}: {e}. Check formatting.")
                except Exception as e:
                     raise ValueError(f"Error processing column '{col}': {e}")

            # --- Calculate Metrics --- 
            new_df['sentiment_score'] = new_df['content'].apply(lambda x: sentiment_analyzer.get_ensemble_sentiment(x)['ensemble_score'] if pd.notna(x) else 0)
            
            # Calculate comment sentiments and contradiction score per post
            comment_sentiments_list = []
            contradiction_scores = []
            for _, row in new_df.iterrows():
                post_comments = row['comments']
                sentiments = []
                if isinstance(post_comments, list) and post_comments:
                    # Use a helper function in SentimentAnalyzer for category mapping
                    sentiments = [sentiment_analyzer.get_sentiment_category(c) for c in post_comments]
                    negative_comments = sum(1 for s in sentiments if s == 'negative')
                    skeptical_comments = sum(1 for s in sentiments if s == 'skeptical')
                    total_comments = len(sentiments)
                    # Weighted contradiction based on negative/skeptical comments
                    contradiction = (0.6 * (negative_comments / total_comments) + 0.4 * (skeptical_comments / total_comments)) if total_comments > 0 else 0
                else:
                    contradiction = 0 # No comments, no contradiction
                comment_sentiments_list.append(sentiments)
                contradiction_scores.append(contradiction)
            
            new_df['comment_sentiments'] = comment_sentiments_list
            new_df['contradiction_score'] = contradiction_scores

            # --- Update Global DataFrame and Filters --- 
            df = new_df # Replace global df with the newly processed data
            company_options = [{'label': c, 'value': c} for c in df['company'].unique()]
            platform_options = [{'label': p, 'value': p} for p in df['platform'].unique()]
            first_company = df['company'].unique()[0]
            first_platform = df['platform'].unique()[0]
            min_date = df['timestamp'].min()
            max_date = df['timestamp'].max()

            status_message = dbc.Alert(f"Successfully processed '{filename}' with {len(df)} rows.", color="success")
            
        else:
             raise ValueError("Invalid file type. Please upload a CSV file.")

    except Exception as e:
        # logger.error(f"Error processing uploaded file: {e}") # Already defined in contradiction_detector
        print(f"Error processing uploaded file: {e}") # Print error for debugging
        status_message = dbc.Alert(f"Error processing file: {str(e)}", color="danger")
        # Return empty options and default values on error
        company_options, first_company = [], None
        platform_options, first_platform = [], None
        min_date, max_date = None, None
        
    return status_message, company_options, first_company, platform_options, first_platform, min_date, max_date


# --- Helper Function for Filtering --- 
def filter_data(company, platform, start_date, end_date):
    filtered = df.copy()
    if company:
        filtered = filtered[filtered['company'] == company]
    if platform:
        filtered = filtered[filtered['platform'] == platform]
    
    # Ensure timestamp column is datetime
    if not pd.api.types.is_datetime64_any_dtype(filtered['timestamp']):
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
        return go.Figure(layout={
            'title': 'Sentiment Trend Over Time',
            'xaxis': {'visible': False}, 'yaxis': {'visible': False},
            'annotations': [{'text': 'No data for selected filters', 'showarrow': False}]
        })
        
    # Calculate rolling average
    filtered_df['sentiment_rolling'] = filtered_df['sentiment_score'].rolling(window=3, min_periods=1, center=True).mean()
    
    fig = px.line(
        filtered_df,
        x='timestamp',
        y=['sentiment_score', 'sentiment_rolling'],
        title='Sentiment Trend Over Time',
        labels={'timestamp': 'Date', 'value': 'Sentiment Score (-1 to 1)'},
        # template='plotly_white' # Set globally now
    )
    fig.update_layout(legend_title_text='Score Type')
    fig.data[0].name = 'Actual Score'
    fig.data[1].name = 'Rolling Avg (3 posts)'
    fig.update_traces(hovertemplate='Date: %{x}<br>Score: %{y:.2f}')
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

# Callback for greenwashing risk
@app.callback(
    Output('greenwashing-risk', 'figure'),
    [Input('company-filter', 'value'),
     Input('platform-filter', 'value'),
     Input('date-range', 'start_date'),
     Input('date-range', 'end_date')]
)
def update_greenwashing_risk(company, platform, start_date, end_date):
    filtered_df = filter_data(company, platform, start_date, end_date)
    
    if filtered_df.empty:
        return go.Figure(layout={
            'title': 'Greenwashing Risk Analysis',
            'xaxis': {'visible': False}, 'yaxis': {'visible': False},
            'annotations': [{'text': 'No data for selected filters', 'showarrow': False}]
        })

    # Calculate risk score per post
    # Risk = (|Sentiment| + Contradiction + CommentRisk) / 3
    # CommentRisk = (NegativeComments + 0.5 * SkepticalComments) / TotalComments
    risk_scores = []
    hover_texts = []
    valid_indices = []

    for idx, row in filtered_df.iterrows():
        sentiment = row['sentiment_score']
        contradiction = row['contradiction_score'] # Already calculated based on comments
        comments_list = row['comment_sentiments'] # List of 'positive', 'negative', 'skeptical'
        
        if not isinstance(comments_list, list) or not comments_list:
            comment_risk = 0
        else:
            negative_comments = sum(1 for s in comments_list if s == 'negative')
            skeptical_comments = sum(1 for s in comments_list if s == 'skeptical')
            total_comments = len(comments_list)
            comment_risk = (negative_comments + 0.5 * skeptical_comments) / total_comments if total_comments > 0 else 0
            
        # Ensure components are numeric
        if pd.isna(sentiment) or pd.isna(contradiction) or pd.isna(comment_risk):
             risk_score = np.nan # Mark as NaN if any component is missing
        else:
            risk_score = (abs(sentiment) + contradiction + comment_risk) / 3
        
        risk_scores.append(risk_score)
        hover_texts.append(f"Post ID: {row['post_id']}<br>Content: {row['content'][:50]}...<br>Risk Score: {risk_score:.2f}")
        if not pd.isna(risk_score):
            valid_indices.append(idx)

    filtered_df['risk_score'] = risk_scores
    valid_df = filtered_df.loc[valid_indices]

    if valid_df.empty:
        return go.Figure(layout={
            'title': 'Greenwashing Risk Analysis',
            'xaxis': {'visible': False}, 'yaxis': {'visible': False},
            'annotations': [{'text': 'No valid risk scores to plot', 'showarrow': False}]
        })
    
    # Normalize size (handle case with single point or all same risk)
    min_risk = valid_df['risk_score'].min()
    max_risk = valid_df['risk_score'].max()
    if max_risk == min_risk:
        sizes = np.full(len(valid_df), 15) # Default size if no variation
    else:
        sizes = 5 + 20 * (valid_df['risk_score'] - min_risk) / (max_risk - min_risk)
    
    fig = px.scatter(
        valid_df,
        x='sentiment_score',
        y='contradiction_score',
        size=sizes,
        color='risk_score',
        color_continuous_scale='RdYlGn_r', # Red (high risk) to Green (low risk)
        range_color=[0, 1], # Risk score range is 0-1
        title='Greenwashing Risk Analysis (Sentiment vs. Contradiction)',
        labels={'sentiment_score': 'Post Sentiment Score', 'contradiction_score': 'Contradiction Score (from Comments)'},
        hover_data={'post_id': True, 'content': True, 'risk_score': ':.2f', 'sentiment_score':':.2f', 'contradiction_score':':.2f'},
        # template='plotly_white' # Set globally now
    )
    
    fig.update_layout(
        coloraxis_colorbar=dict(title="Risk Score"),
        xaxis=dict(range=[-1.1, 1.1]),
        yaxis=dict(range=[-0.1, 1.1])
    )
    fig.update_traces(marker=dict(sizemin=5, sizeref=max(sizes)/(25**2) if max(sizes)>0 else 1), selector=dict(type='scatter')) # Adjust sizeref for proper scaling

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
    
    # Aggregate by date (e.g., daily)
    daily_comments = filtered_df.set_index('timestamp').resample('D')['comment_count'].sum().reset_index()
    daily_comments['rolling_avg'] = daily_comments['comment_count'].rolling(window=7, min_periods=1, center=True).mean()

    if daily_comments.empty:
         return go.Figure(layout={
            'title': 'Comment Volume Over Time',
            'xaxis': {'visible': False}, 'yaxis': {'visible': False},
            'annotations': [{'text': 'No comments to plot', 'showarrow': False}]
        })
        
    fig = px.line(
        daily_comments,
        x='timestamp',
        y=['comment_count', 'rolling_avg'],
        title='Comment Volume Over Time',
        labels={'timestamp': 'Date', 'value': 'Number of Comments'},
        # template='plotly_white' # Set globally now
    )
    fig.data[0].name = 'Daily Count'
    fig.data[1].name = 'Rolling Avg (7 days)'
    fig.update_layout(legend_title_text='Metric')
    fig.update_traces(hovertemplate='Date: %{x}<br>Comments: %{y:.0f}')
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
    [Input('company-filter', 'value')] # Trigger only when company changes
)
def update_company_risk(company):
    if not company:
        return "--", "Select a company"
    
    company_df = df[df['company'] == company].copy()
    if company_df.empty:
         return "N/A", "No data for this company"

    # Calculate company-level risk (example: average of post risk scores)
    # Ensure 'risk_score' exists from the greenwashing plot calculation or recalculate
    if 'risk_score' not in company_df.columns:
         # Simplified recalculation if needed (consider efficiency for large data)
         risk_scores = []
         for _, row in company_df.iterrows():
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
             risk_scores.append(risk_score)
         company_df['risk_score'] = risk_scores
         
    # Calculate average risk, ignoring NaNs
    avg_company_risk = company_df['risk_score'].mean(skipna=True)
    
    if pd.isna(avg_company_risk):
        return "N/A", "Could not calculate risk score"

    risk_level = "Low"
    risk_color = "success"
    if avg_company_risk > 0.7:
        risk_level = "High"
        risk_color = "danger"
    elif avg_company_risk > 0.4:
        risk_level = "Medium"
        risk_color = "warning"
        
    score_display = f"{avg_company_risk:.2f}"
    description = html.Span(f"{risk_level} Risk", className=f"badge bg-{risk_color}")

    return score_display, description

# Run the app
if __name__ == '__main__':
    app.run(debug=True) 