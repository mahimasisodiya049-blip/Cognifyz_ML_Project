import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class RestaurantEngine:
    def __init__(self, df):
        self.df = df.copy()
        self._preprocess_data()
        self.prepare_recommender()

    def _preprocess_data(self):
        """Ensures all data types are correct for ML operations."""
        self.df.columns = self.df.columns.str.strip()
        num_cols = ['Aggregate rating', 'Votes', 'Average Cost for two', 'Price range', 'Latitude', 'Longitude']
        for col in num_cols:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce').fillna(0.0)
        self.df['Cuisines'] = self.df['Cuisines'].fillna('Other')

    def train_rating_model(self):
        """Trains the Random Forest model and returns performance metrics."""
        features = ['Price range', 'Votes', 'Average Cost for two']
        X = self.df[features]
        y = self.df['Aggregate rating']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.model.fit(X_train, y_train)
        
        score = self.model.score(X_test, y_test)
        
        # Data for the Prediction History chart
        history_df = pd.DataFrame({
            'Actual Rating': y_test.head(15).values,
            'Predicted Rating': self.model.predict(X_test.head(15))
        })
        return score, history_df

    def predict_rating(self, price, votes, cost):
        """Predicts a single rating based on user input."""
        if hasattr(self, 'model'):
            return self.model.predict([[price, votes, cost]])[0]
        return 0.0

    def prepare_recommender(self):
        """Builds the TF-IDF matrix for cuisine-based similarity."""
        tfidf = TfidfVectorizer(stop_words='english')
        tfidf_matrix = tfidf.fit_transform(self.df['Cuisines'].astype(str))
        self.cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

    def get_smart_recommendations(self, name, top_n=5):
        """Returns the top N similar restaurants based on cosine similarity."""
        try:
            idx = self.df[self.df['Restaurant Name'] == name].index[0]
            sim_scores = sorted(list(enumerate(self.cosine_sim[idx])), key=lambda x: x[1], reverse=True)
            indices = [i[0] for i in sim_scores[1:top_n+1]]
            return self.df.iloc[indices][['Restaurant Name', 'Cuisines', 'Aggregate rating', 'City']]
        except:
            return None