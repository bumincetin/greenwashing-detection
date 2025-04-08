import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import json

class DataGenerator:
    def __init__(self):
        self.platforms = ['Twitter', 'LinkedIn', 'Facebook', 'Instagram']
        self.companies = [
            'GreenTech Solutions',  # Low risk - transparent and consistent
            'EcoFriendly Corp',    # Low risk - moderate claims with evidence
            'Sustainable Industries', # Medium risk - some inconsistencies
            'Clean Energy Systems',  # Medium risk - ambitious claims
            'GreenWash Inc',        # High risk - exaggerated claims
            'EcoMarketing Pro',     # High risk - misleading statements
            'GreenFuture Ltd',      # High risk - contradictory claims
            'Sustainable PR',       # High risk - vague promises
            'CleanTech Global',     # Medium risk - mixed messaging
            'EcoSolutions Plus'     # Medium risk - inconsistent reporting
        ]
        
        # Company risk profiles
        self.company_risk_profiles = {
            'GreenTech Solutions': {
                'risk_level': 'low',
                'sentiment_range': (-0.2, 0.3),
                'contradiction_prob': 0.1,
                'comment_ratio': {'positive': 0.7, 'skeptical': 0.2, 'negative': 0.1}
            },
            'EcoFriendly Corp': {
                'risk_level': 'low',
                'sentiment_range': (-0.3, 0.4),
                'contradiction_prob': 0.15,
                'comment_ratio': {'positive': 0.6, 'skeptical': 0.3, 'negative': 0.1}
            },
            'Sustainable Industries': {
                'risk_level': 'medium',
                'sentiment_range': (-0.4, 0.5),
                'contradiction_prob': 0.3,
                'comment_ratio': {'positive': 0.5, 'skeptical': 0.3, 'negative': 0.2}
            },
            'Clean Energy Systems': {
                'risk_level': 'medium',
                'sentiment_range': (-0.5, 0.6),
                'contradiction_prob': 0.35,
                'comment_ratio': {'positive': 0.4, 'skeptical': 0.4, 'negative': 0.2}
            },
            'GreenWash Inc': {
                'risk_level': 'high',
                'sentiment_range': (-0.7, 0.8),
                'contradiction_prob': 0.6,
                'comment_ratio': {'positive': 0.3, 'skeptical': 0.4, 'negative': 0.3}
            },
            'EcoMarketing Pro': {
                'risk_level': 'high',
                'sentiment_range': (-0.8, 0.9),
                'contradiction_prob': 0.65,
                'comment_ratio': {'positive': 0.2, 'skeptical': 0.5, 'negative': 0.3}
            },
            'GreenFuture Ltd': {
                'risk_level': 'high',
                'sentiment_range': (-0.9, 1.0),
                'contradiction_prob': 0.7,
                'comment_ratio': {'positive': 0.1, 'skeptical': 0.5, 'negative': 0.4}
            },
            'Sustainable PR': {
                'risk_level': 'high',
                'sentiment_range': (-1.0, 1.0),
                'contradiction_prob': 0.75,
                'comment_ratio': {'positive': 0.1, 'skeptical': 0.4, 'negative': 0.5}
            },
            'CleanTech Global': {
                'risk_level': 'medium',
                'sentiment_range': (-0.5, 0.7),
                'contradiction_prob': 0.4,
                'comment_ratio': {'positive': 0.3, 'skeptical': 0.5, 'negative': 0.2}
            },
            'EcoSolutions Plus': {
                'risk_level': 'medium',
                'sentiment_range': (-0.6, 0.6),
                'contradiction_prob': 0.45,
                'comment_ratio': {'positive': 0.3, 'skeptical': 0.4, 'negative': 0.3}
            }
        }
        
        # Sample sustainability-related content with risk levels
        self.sustainability_content = {
            'low': [
                "Our facilities now run on 100% renewable energy, verified by third-party audits.",
                "We've achieved a 40% reduction in carbon emissions, with detailed reporting available.",
                "Our new sustainable packaging has been certified by environmental standards.",
                "Transparency is key: Here's our detailed environmental impact assessment.",
                "We're committed to reducing our carbon footprint with measurable goals."
            ],
            'medium': [
                "Exciting news! We're launching our new green initiative to reduce emissions.",
                "Our commitment to sustainability continues with ambitious goals for 2025.",
                "We're making progress on our environmental goals. Stay tuned for updates!",
                "Join us in our journey towards a greener future with innovative solutions.",
                "Our new eco-friendly product line demonstrates our environmental commitment."
            ],
            'high': [
                "Revolutionary breakthrough! Our new technology will solve all environmental problems!",
                "We're leading the way in sustainability! No other company comes close!",
                "Our green credentials are unmatched! Trust us, we're the most sustainable!",
                "The future is green, and we're already there! Join the revolution!",
                "Our innovative solution will save the planet! No questions asked!"
            ]
        }
        
        # Sample comment templates for different scenarios
        self.comment_templates = {
            'positive': [
                "Great initiative! This shows real commitment to sustainability.",
                "Impressive steps towards a greener future!",
                "This is exactly what we need more companies to do.",
                "Well done on taking concrete action!",
                "This makes me proud to be a customer."
            ],
            'skeptical': [
                "I'll believe it when I see the actual results.",
                "How do we know this isn't just marketing?",
                "Show us the data to back these claims.",
                "Actions speak louder than words.",
                "Let's see if this leads to real change."
            ],
            'negative': [
                "This feels like greenwashing to me.",
                "Another PR stunt, nothing more.",
                "Where's the transparency in your claims?",
                "Talk is cheap, show us real action.",
                "This seems too good to be true."
            ]
        }
        
        # Sample engagement metrics ranges
        self.engagement_ranges = {
            'Twitter': {'likes': (10, 1000), 'retweets': (5, 500), 'replies': (1, 100)},
            'LinkedIn': {'likes': (20, 2000), 'shares': (5, 500), 'comments': (2, 200)},
            'Facebook': {'likes': (30, 3000), 'shares': (10, 1000), 'comments': (5, 500)},
            'Instagram': {'likes': (50, 5000), 'comments': (5, 500), 'saves': (10, 1000)}
        }
        
        self.sustainability_keywords = [
            "sustainable", "green", "eco-friendly", "renewable", "carbon-neutral",
            "environmental", "climate", "emissions", "recycling", "clean energy"
        ]
    
    def generate_timestamp(self, days_back=30):
        """Generate a random timestamp within the specified range"""
        random_days = random.randint(0, days_back)
        random_hours = random.randint(0, 24)
        random_minutes = random.randint(0, 60)
        return datetime.now() - timedelta(days=random_days, hours=random_hours, minutes=random_minutes)
    
    def generate_engagement_metrics(self, platform):
        """Generate realistic engagement metrics for a platform"""
        ranges = self.engagement_ranges[platform]
        metrics = {}
        
        if platform == 'Twitter':
            metrics = {
                'likes': random.randint(*ranges['likes']),
                'retweets': random.randint(*ranges['retweets']),
                'replies': random.randint(*ranges['replies'])
            }
        elif platform == 'LinkedIn':
            metrics = {
                'likes': random.randint(*ranges['likes']),
                'shares': random.randint(*ranges['shares']),
                'comments': random.randint(*ranges['comments'])
            }
        elif platform == 'Facebook':
            metrics = {
                'likes': random.randint(*ranges['likes']),
                'shares': random.randint(*ranges['shares']),
                'comments': random.randint(*ranges['comments'])
            }
        elif platform == 'Instagram':
            metrics = {
                'likes': random.randint(*ranges['likes']),
                'comments': random.randint(*ranges['comments']),
                'saves': random.randint(*ranges['saves'])
            }
        
        return metrics
    
    def generate_comments(self, post_content, num_comments, comment_ratio):
        """Generate realistic comments for a post with specified sentiment ratio"""
        comments = []
        comment_sentiments = []
        
        for _ in range(num_comments):
            # Determine comment sentiment based on specified ratio
            rand = random.random()
            if rand < comment_ratio['positive']:
                sentiment = 'positive'
            elif rand < comment_ratio['positive'] + comment_ratio['skeptical']:
                sentiment = 'skeptical'
            else:
                sentiment = 'negative'
            
            # Generate comment
            comment = random.choice(self.comment_templates[sentiment])
            comments.append(comment)
            comment_sentiments.append(sentiment)
        
        return comments, comment_sentiments
    
    def generate_synthetic_data(self, num_posts=100):
        """Generate synthetic social media data"""
        data = []
        
        for i in range(num_posts):
            company = random.choice(self.companies)
            platform = random.choice(self.platforms)
            timestamp = self.generate_timestamp()
            
            # Get company risk profile
            risk_profile = self.company_risk_profiles[company]
            risk_level = risk_profile['risk_level']
            
            # Generate content based on risk level
            content = random.choice(self.sustainability_content[risk_level])
            
            # Generate sentiment score based on risk profile
            sentiment_range = risk_profile['sentiment_range']
            sentiment_score = random.uniform(sentiment_range[0], sentiment_range[1])
            
            # Generate contradiction score based on risk profile
            contradiction_score = random.uniform(0, 1) if random.random() < risk_profile['contradiction_prob'] else random.uniform(0, 0.3)
            
            # Generate engagement metrics
            engagement_metrics = self.generate_engagement_metrics(platform)
            
            # Generate comments with appropriate sentiment distribution
            num_comments = random.randint(3, 8)
            comments, comment_sentiments = self.generate_comments(content, num_comments, risk_profile['comment_ratio'])
            
            # Calculate post-level risk score
            post_risk_score = (abs(sentiment_score) + contradiction_score + 
                             (sum(1 for s in comment_sentiments if s == 'negative') + 
                              0.5 * sum(1 for s in comment_sentiments if s == 'skeptical')) / num_comments) / 3
            
            post_data = {
                'post_id': i + 1,
                'company': company,
                'platform': platform,
                'timestamp': timestamp,
                'content': content,
                'engagement_metrics': json.dumps(engagement_metrics),
                'comments': json.dumps(comments),
                'sentiment_score': sentiment_score,
                'contradiction_score': contradiction_score,
                'comment_sentiments': json.dumps(comment_sentiments),
                'risk_level': risk_level,
                'post_risk_score': post_risk_score
            }
            
            data.append(post_data)
        
        # Create DataFrame
        df = pd.DataFrame(data)
        
        # Calculate company-level greenwashing risk scores
        company_metrics = {}
        for company in self.companies:
            company_posts = df[df['company'] == company]
            
            # Calculate weighted average of post risk scores
            avg_post_risk = company_posts['post_risk_score'].mean()
            
            # Calculate sentiment volatility
            sentiment_std = company_posts['sentiment_score'].std()
            
            # Calculate comment controversy ratio
            total_comments = sum(len(json.loads(comments)) for comments in company_posts['comments'])
            negative_comments = sum(
                sum(1 for s in json.loads(sentiments) if s == "negative")
                for sentiments in company_posts['comment_sentiments']
            )
            skeptical_comments = sum(
                sum(1 for s in json.loads(sentiments) if s == "skeptical")
                for sentiments in company_posts['comment_sentiments']
            )
            controversy_ratio = (negative_comments + 0.5 * skeptical_comments) / total_comments
            
            # Calculate final company risk score (0-1)
            company_risk_score = (
                0.4 * avg_post_risk +  # Average post risk
                0.3 * sentiment_std +  # Sentiment volatility
                0.3 * controversy_ratio  # Comment controversy
            )
            
            company_metrics[company] = {
                'avg_post_risk': avg_post_risk,
                'sentiment_volatility': sentiment_std,
                'controversy_ratio': controversy_ratio,
                'company_risk_score': company_risk_score
            }
        
        # Add company metrics to the DataFrame
        df['company_risk_score'] = df['company'].map(
            lambda x: company_metrics[x]['company_risk_score']
        )
        
        return df

    def save_to_csv(self, df, filename='synthetic_data.csv'):
        df.to_csv(filename, index=False)
        print(f"Generated {len(df)} posts with varying risk levels")
        print("\nCompany-level greenwashing risk scores:")
        for company in self.companies:
            company_data = df[df['company'] == company].iloc[0]
            print(f"{company}: {company_data['company_risk_score']:.3f}") 