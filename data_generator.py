import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

class DataGenerator:
    def __init__(self):
        self.platforms = ['Twitter', 'LinkedIn', 'Facebook', 'Instagram']
        self.companies = [
            'EcoTech Solutions', 'Green Energy Corp', 'Sustainable Industries',
            'Clean Power Systems', 'EcoFriendly Products'
        ]
        
        # Sample sustainability-related content
        self.sustainability_content = [
            "We're proud to announce our new carbon-neutral initiative!",
            "Our facilities now run on 100% renewable energy.",
            "We've achieved zero waste in our manufacturing process.",
            "Join us in our journey towards sustainability.",
            "Our new eco-friendly product line is now available!",
            "We're committed to reducing our carbon footprint.",
            "Our sustainability report shows great progress.",
            "We've planted 1 million trees this year.",
            "Our packaging is now 100% recyclable.",
            "We're investing in clean energy technology."
        ]
        
        # Sample engagement metrics ranges
        self.engagement_ranges = {
            'Twitter': {'likes': (10, 1000), 'retweets': (5, 500), 'replies': (1, 100)},
            'LinkedIn': {'likes': (20, 2000), 'shares': (5, 500), 'comments': (2, 200)},
            'Facebook': {'likes': (30, 3000), 'shares': (10, 1000), 'comments': (5, 500)},
            'Instagram': {'likes': (50, 5000), 'comments': (5, 500), 'saves': (10, 1000)}
        }
    
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
    
    def generate_synthetic_data(self, num_posts=100):
        """Generate synthetic social media data"""
        data = []
        
        for i in range(num_posts):
            company = random.choice(self.companies)
            platform = random.choice(self.platforms)
            timestamp = self.generate_timestamp()
            content = random.choice(self.sustainability_content)
            engagement = self.generate_engagement_metrics(platform)
            
            # Generate post ID
            post_id = f"{company[:3].upper()}{platform[:2].upper()}{i:04d}"
            
            # Generate comment ID (some posts might not have comments)
            comment_id = f"CMT{i:04d}" if random.random() < 0.7 else None
            
            # Generate synthetic sentiment and contradiction scores
            sentiment_score = random.uniform(-1, 1)
            contradiction_score = random.uniform(0, 1)
            
            post_data = {
                'post_id': post_id,
                'company': company,
                'platform': platform,
                'timestamp': timestamp,
                'content': content,
                'engagement_metrics': engagement,
                'comment_id': comment_id,
                'sentiment_score': sentiment_score,
                'contradiction_score': contradiction_score
            }
            
            data.append(post_data)
        
        return pd.DataFrame(data)
    
    def save_to_csv(self, df, filename='synthetic_social_media_data.csv'):
        """Save the generated data to a CSV file"""
        df.to_csv(filename, index=False)
        return filename 