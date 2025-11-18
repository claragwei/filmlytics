import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
import certifi
import os
warnings.filterwarnings('ignore')

# Create output folder
output_folder = 'visualizations'
os.makedirs(output_folder, exist_ok=True)

# Connect to MongoDB
print("Connecting to MongoDB...")
client = MongoClient(
    "mongodb+srv://cinemaniacs:filmlytics@filmlytics.1emhcue.mongodb.net/?appName=filmlytics",
    server_api=ServerApi('1'),
    tlsCAFile=certifi.where()
)

db = client['cinemaniacs']
collection = db['movies']

# Load data
print("Loading data from MongoDB...")
data = list(collection.find())
print(f"Loaded {len(data):,} movies")

# Flatten nested structure into DataFrame
records = []
for doc in data:
    record = {
        'tmdb_id': doc.get('tmdb_id'),
        'title': doc.get('title'),
        
        # Release info
        'release_date': doc.get('release_info', {}).get('tmdb_release_date'),
        'days_until_release': doc.get('release_info', {}).get('days_until_release'),
        
        # Production
        'budget': doc.get('production', {}).get('budget', 0),
        'runtime': doc.get('production', {}).get('runtime'),
        'genres': doc.get('production', {}).get('genres', []),
        'production_companies': doc.get('production', {}).get('production_companies', []),
        'production_countries': doc.get('production', {}).get('production_countries', []),
        
        # People
        'cast': doc.get('people', {}).get('cast', []),
        'directors': doc.get('people', {}).get('directors', []),
        
        # TMDB metrics
        'vote_count': doc.get('tmdb_metrics', {}).get('vote_count'),
        'vote_average': doc.get('tmdb_metrics', {}).get('vote_average'),
        'is_successful': doc.get('tmdb_metrics', {}).get('is_successful'),
        
        # Rotten Tomatoes
        'has_rt_url': doc.get('rotten_tomatoes', {}).get('has_rt_url', False),
        'critic_score': doc.get('rotten_tomatoes', {}).get('critic_score'),
        'audience_score': doc.get('rotten_tomatoes', {}).get('audience_score'),
        
        # Sentiment
        'description_sentiment': doc.get('sentiment', {}).get('description_sentiment_score'),
        
        # Trailer metrics
        'view_count': doc.get('trailer', {}).get('metrics', {}).get('view_count'),
        'like_count': doc.get('trailer', {}).get('metrics', {}).get('like_count'),
        'comment_count': doc.get('trailer', {}).get('metrics', {}).get('comment_count'),
        'trailer_published_at': doc.get('trailer', {}).get('published_at'),
        'is_official_trailer': doc.get('trailer', {}).get('official'),
    }
    records.append(record)

df = pd.DataFrame(records)
print(f"Flattened to {len(df.columns)} columns")

# Data preprocessing
df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
df['trailer_published_at'] = pd.to_datetime(df['trailer_published_at'], errors='coerce')

# Feature engineering
print("\nFeature engineering...")

# Basic features
df['budget_log'] = np.log1p(df['budget'])
df['has_budget'] = (df['budget'] > 0).astype(int)
df['has_runtime'] = df['runtime'].notna().astype(int)

# Temporal features
df['release_year'] = df['release_date'].dt.year
df['release_month'] = df['release_date'].dt.month
df['release_quarter'] = df['release_date'].dt.quarter
df['is_summer_release'] = df['release_month'].isin([6, 7, 8]).astype(int)
df['is_holiday_release'] = df['release_month'].isin([11, 12]).astype(int)

# Genre features
df['genre_count'] = df['genres'].apply(lambda x: len(x) if isinstance(x, list) else 0)
df['has_genre'] = (df['genre_count'] > 0).astype(int)

# Get top genres
all_genres = []
for genres in df['genres']:
    if isinstance(genres, list):
        all_genres.extend(genres)
top_genres = pd.Series(all_genres).value_counts().head(5).index.tolist()

for genre in top_genres:
    df[f'is_{genre.lower().replace(" ", "_")}'] = df['genres'].apply(
        lambda x: 1 if isinstance(x, list) and genre in x else 0
    )

# People features
df['cast_count'] = df['cast'].apply(lambda x: len(x) if isinstance(x, list) else 0)
df['director_count'] = df['directors'].apply(lambda x: len(x) if isinstance(x, list) else 0)
df['has_cast'] = (df['cast_count'] > 0).astype(int)
df['has_director'] = (df['director_count'] > 0).astype(int)

# Production features
df['production_company_count'] = df['production_companies'].apply(lambda x: len(x) if isinstance(x, list) else 0)
df['production_country_count'] = df['production_countries'].apply(lambda x: len(x) if isinstance(x, list) else 0)
df['is_us_production'] = df['production_countries'].apply(
    lambda x: 1 if isinstance(x, list) and 'United States of America' in x else 0
)

# Sentiment features (expanded)
df['has_sentiment'] = df['description_sentiment'].notna().astype(int)
df['sentiment_positive'] = (df['description_sentiment'] > 0).astype(int)
df['sentiment_negative'] = (df['description_sentiment'] < 0).astype(int)

# Sentiment magnitude and strength
df['sentiment_magnitude'] = np.abs(df['description_sentiment'].fillna(0))
df['sentiment_strength'] = df['description_sentiment'].fillna(0) ** 2

# Sentiment categories
df['very_positive'] = (df['description_sentiment'] > 0.5).astype(int)
df['very_negative'] = (df['description_sentiment'] < -0.5).astype(int)
df['neutral_sentiment'] = (np.abs(df['description_sentiment'].fillna(0)) < 0.1).astype(int)
df['moderate_positive'] = ((df['description_sentiment'] > 0.1) & (df['description_sentiment'] <= 0.5)).astype(int)
df['moderate_negative'] = ((df['description_sentiment'] < -0.1) & (df['description_sentiment'] >= -0.5)).astype(int)

# Trailer features
df['has_trailer_data'] = df['view_count'].notna().astype(int)
df['view_count'] = df['view_count'].fillna(0)
df['like_count'] = df['like_count'].fillna(0)
df['comment_count'] = df['comment_count'].fillna(0)

# Engagement rates
df['like_rate'] = np.where(df['view_count'] > 0, df['like_count'] / df['view_count'], 0)
df['comment_rate'] = np.where(df['view_count'] > 0, df['comment_count'] / df['view_count'], 0)

# Rotten Tomatoes features
df['has_critic_score'] = df['critic_score'].notna().astype(int)
df['has_audience_score'] = df['audience_score'].notna().astype(int)

# Binning features
print("\nCreating binned features...")

# Budget categories
budget_bins = [0, 1e6, 10e6, 50e6, 100e6, np.inf]
budget_labels = ['micro', 'low', 'medium', 'high', 'blockbuster']
df['budget_tier'] = pd.cut(df['budget'], bins=budget_bins, labels=budget_labels)

for tier in budget_labels:
    df[f'budget_{tier}'] = (df['budget_tier'] == tier).astype(int)

# Runtime categories
runtime_bins = [0, 80, 100, 120, 150, np.inf]
runtime_labels = ['short', 'standard', 'long', 'very_long', 'marathon']
df['runtime_category'] = pd.cut(df['runtime'].fillna(0), bins=runtime_bins, labels=runtime_labels)

for cat in runtime_labels:
    df[f'runtime_{cat}'] = (df['runtime_category'] == cat).astype(int)

# Engagement tiers (only for movies with trailer data)
df['engagement_tier'] = 'none'
has_views = df['view_count'] > 0
if has_views.sum() > 0:
    view_quantiles = df.loc[has_views, 'view_count'].quantile([0.2, 0.4, 0.6, 0.8])
    df.loc[has_views, 'engagement_tier'] = pd.cut(
        df.loc[has_views, 'view_count'],
        bins=[0] + view_quantiles.tolist() + [np.inf],
        labels=['very_low', 'low', 'medium', 'high', 'viral']
    ).astype(str)

for tier in ['none', 'very_low', 'low', 'medium', 'high', 'viral']:
    df[f'engagement_{tier}'] = (df['engagement_tier'] == tier).astype(int)

# Release year bins (by decade)
df['release_decade'] = (df['release_year'] // 10) * 10
decade_dummies = pd.get_dummies(df['release_decade'], prefix='decade')
df = pd.concat([df, decade_dummies], axis=1)

# Extract all production companies
all_companies = []
for companies in df['production_companies']:
    if isinstance(companies, list):
        all_companies.extend(companies)

# Get top 15 production companies by frequency
top_companies = pd.Series(all_companies).value_counts().head(15)
print(f"  Top 15 production companies identified")

# Create dummy variables for top companies
for company in top_companies.index:
    # Create safe column name
    safe_name = company.lower().replace(' ', '_').replace('.', '').replace('-', '_')
    safe_name = ''.join(c for c in safe_name if c.isalnum() or c == '_')[:40]  # Limit length
    col_name = f'company_{safe_name}'

    df[col_name] = df['production_companies'].apply(
        lambda x: 1 if isinstance(x, list) and company in x else 0
    )

# Time decay features
print("\nCreating time decay features...")

def exponential_decay(value, days_before_release, half_life=30):
    """Apply exponential decay based on days before release"""
    if pd.isna(value) or pd.isna(days_before_release) or days_before_release <= 0:
        return 0
    decay_rate = np.log(2) / half_life
    return value * np.exp(-decay_rate * days_before_release)

def recency_weight(days_before_release, peak_days=14):
    """Give more weight to trailers released near optimal timing"""
    if pd.isna(days_before_release):
        return 0
    return np.exp(-((days_before_release - peak_days) ** 2) / (2 * (peak_days ** 2)))

# Apply time decay only to movies with trailer data
df['views_exp_decay_30'] = df.apply(
    lambda row: exponential_decay(row['view_count'], row['days_until_release'], 30)
    if row['has_trailer_data'] and pd.notna(row['days_until_release']) else 0, axis=1
)

df['likes_exp_decay_30'] = df.apply(
    lambda row: exponential_decay(row['like_count'], row['days_until_release'], 30)
    if row['has_trailer_data'] and pd.notna(row['days_until_release']) else 0, axis=1
)

df['views_recency_weighted'] = df.apply(
    lambda row: row['view_count'] * recency_weight(row['days_until_release'], 14)
    if row['has_trailer_data'] and pd.notna(row['days_until_release']) else 0, axis=1
)

df['likes_recency_weighted'] = df.apply(
    lambda row: row['like_count'] * recency_weight(row['days_until_release'], 14)
    if row['has_trailer_data'] and pd.notna(row['days_until_release']) else 0, axis=1
)

df['views_per_day'] = df.apply(
    lambda row: row['view_count'] / (np.abs(row['days_until_release']) + 1)
    if row['has_trailer_data'] and pd.notna(row['days_until_release']) and row['days_until_release'] != 0 else 0, axis=1
)

df['likes_per_day'] = df.apply(
    lambda row: row['like_count'] / (np.abs(row['days_until_release']) + 1)
    if row['has_trailer_data'] and pd.notna(row['days_until_release']) and row['days_until_release'] != 0 else 0, axis=1
)

print(f"Movies with trailer data: {df['has_trailer_data'].sum():,} ({df['has_trailer_data'].mean()*100:.1f}%)")
print(f"Movies with RT scores: {df['has_rt_url'].sum():,} ({df['has_rt_url'].mean()*100:.1f}%)")

# Select features for modeling
feature_columns = [
    # Basic
    'budget', 'budget_log', 'has_budget', 'runtime', 'has_runtime',

    # Temporal
    'release_year', 'release_month', 'release_quarter',
    'is_summer_release', 'is_holiday_release',

    # Genre
    'genre_count', 'has_genre',

    # People
    'cast_count', 'director_count', 'has_cast', 'has_director',

    # Production
    'production_company_count', 'production_country_count', 'is_us_production',

    # Sentiment (expanded)
    'has_sentiment', 'sentiment_positive', 'sentiment_negative',
    'sentiment_magnitude', 'sentiment_strength',
    'very_positive', 'very_negative', 'neutral_sentiment',
    'moderate_positive', 'moderate_negative',

    # Trailer (original)
    'has_trailer_data', 'view_count', 'like_count', 'comment_count',
    'like_rate', 'comment_rate',

    # Time decay features
    'views_exp_decay_30', 'likes_exp_decay_30',
    'views_recency_weighted', 'likes_recency_weighted',
    'views_per_day', 'likes_per_day',

    # Rotten Tomatoes
    'has_rt_url', 'has_critic_score', 'has_audience_score',
]

# Add genre dummy variables
for genre in top_genres:
    col_name = f'is_{genre.lower().replace(" ", "_")}'
    if col_name in df.columns:
        feature_columns.append(col_name)

# Add budget tier dummies
for tier in ['micro', 'low', 'medium', 'high', 'blockbuster']:
    if f'budget_{tier}' in df.columns:
        feature_columns.append(f'budget_{tier}')

# Add runtime category dummies
for cat in ['short', 'standard', 'long', 'epic', 'marathon']:
    if f'runtime_{cat}' in df.columns:
        feature_columns.append(f'runtime_{cat}')

# Add engagement tier dummies
for tier in ['none', 'very_low', 'low', 'medium', 'high', 'viral']:
    if f'engagement_{tier}' in df.columns:
        feature_columns.append(f'engagement_{tier}')

# Add decade dummies
decade_cols = [col for col in df.columns if col.startswith('decade_')]
feature_columns.extend(decade_cols)

# Add production company dummies
company_cols = [col for col in df.columns if col.startswith('company_')]
feature_columns.extend(company_cols)

# Prepare data first
df_model = df[df['is_successful'].notna() & df['release_date'].notna()].copy()
df_model = df_model.sort_values('release_date').reset_index(drop=True)

# Validate feature columns exist in df_model
valid_feature_columns = []
for col in feature_columns:
    if col in df_model.columns:
        valid_feature_columns.append(col)
    else:
        print(f"Warning: Feature '{col}' not found in dataframe")

feature_columns = valid_feature_columns
print(f"Total features: {len(feature_columns)}")

# Check for duplicate feature names
if len(feature_columns) != len(set(feature_columns)):
    print(f"Warning: Duplicate feature names found!")
    from collections import Counter
    duplicates = [item for item, count in Counter(feature_columns).items() if count > 1]
    print(f"  Duplicates: {duplicates}")
    # Remove duplicates
    feature_columns = list(dict.fromkeys(feature_columns))
    print(f"  After removing duplicates: {len(feature_columns)} features")

X = df_model[feature_columns].copy()
y = df_model['is_successful'].astype(int).copy()  # Convert boolean to int

# Check for non-numeric columns
object_cols = X.select_dtypes(include=['object']).columns.tolist()
if object_cols:
    print(f"Warning: {len(object_cols)} object columns found, converting to numeric:")
    print(f"  {object_cols[:5]}")
    for col in object_cols:
        X[col] = pd.to_numeric(X[col], errors='coerce')

# Fill NaN and convert to float
X = X.fillna(0).astype(float)

# Reset index to avoid any index-related issues
X = X.reset_index(drop=True)
y = y.reset_index(drop=True)

# Random train-test split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTrain: {len(X_train):,} movies")
print(f"Test: {len(X_test):,} movies")

# Check class balance
print(f"\nClass balance:")
print(f"  Successful: {y_train.sum():,} ({y_train.mean():.1%})")
print(f"  Unsuccessful: {(len(y_train) - y_train.sum()):,} ({(1-y_train.mean()):.1%})")

# Calculate scale_pos_weight for class imbalance
scale_pos_weight = (len(y_train) - y_train.sum()) / y_train.sum()
print(f"  Scale pos weight: {scale_pos_weight:.2f}")

# Improved hyperparameter search
print("\nHyperparameter tuning...")

param_distributions = {
    'n_estimators': [400, 500, 600, 700, 800],
    'learning_rate': [0.005, 0.01, 0.02, 0.03, 0.05],
    'max_depth': [4, 5, 6, 7, 8, 9],
    'min_child_weight': [1, 2, 3, 5, 7],
    'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
    'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
    'colsample_bylevel': [0.6, 0.7, 0.8, 0.9],
    'gamma': [0, 0.01, 0.05, 0.1, 0.2],
    'reg_alpha': [0, 0.01, 0.1, 0.5, 1],
    'reg_lambda': [0.5, 1, 1.5, 2, 3]
}

base_model = xgb.XGBClassifier(
    objective='binary:logistic',
    eval_metric='auc',
    scale_pos_weight=scale_pos_weight,  # Handle class imbalance
    random_state=42
)

tscv = TimeSeriesSplit(n_splits=3)

random_search = RandomizedSearchCV(
    estimator=base_model,
    param_distributions=param_distributions,
    n_iter=50,  # Increased from 30
    scoring='roc_auc',
    cv=tscv,
    verbose=1,
    random_state=42,
    n_jobs=-1
)

random_search.fit(X_train, y_train)

print("\nBest hyperparameters:")
for param, value in random_search.best_params_.items():
    print(f"  {param:20s}: {value}")
print(f"Best CV ROC AUC: {random_search.best_score_:.4f}")

# Use best model
model = random_search.best_estimator_
model.fit(X_train, y_train)

# Evaluation
y_pred_test = model.predict(X_test)
y_pred_proba_test = model.predict_proba(X_test)[:, 1]

test_accuracy = accuracy_score(y_test, y_pred_test)
test_auc = roc_auc_score(y_test, y_pred_proba_test)

print(f"\nAccuracy: {test_accuracy:.2%}")
print(f"ROC AUC: {test_auc:.4f}")

# Feature importance
feature_importance = pd.DataFrame({
    'feature': feature_columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 15 Most Important Features:")
print(feature_importance.head(15).to_string(index=False))

# Feature selection - keep features with importance > threshold
importance_threshold = 0.001
selected_features = feature_importance[feature_importance['importance'] > importance_threshold]['feature'].tolist()
removed_features = len(feature_columns) - len(selected_features)

print(f"\nFeature selection:")
print(f"  Original features: {len(feature_columns)}")
print(f"  Selected features: {len(selected_features)} (importance > {importance_threshold})")
print(f"  Removed features: {removed_features}")

# Retrain with selected features if any were removed
if removed_features > 0:
    print("\nRetraining with selected features...")
    X_train_selected = X_train[selected_features]
    X_test_selected = X_test[selected_features]

    model_selected = xgb.XGBClassifier(
        **random_search.best_params_,
        objective='binary:logistic',
        eval_metric='auc',
        scale_pos_weight=scale_pos_weight,
        random_state=42
    )

    model_selected.fit(X_train_selected, y_train)

    y_pred_test_selected = model_selected.predict(X_test_selected)
    y_pred_proba_test_selected = model_selected.predict_proba(X_test_selected)[:, 1]

    test_accuracy_selected = accuracy_score(y_test, y_pred_test_selected)
    test_auc_selected = roc_auc_score(y_test, y_pred_proba_test_selected)

    print(f"\nWith feature selection:")
    print(f"  Accuracy: {test_accuracy_selected:.2%} (vs {test_accuracy:.2%})")
    print(f"  ROC AUC: {test_auc_selected:.4f} (vs {test_auc:.4f})")

    # Use selected model if it's better
    if test_auc_selected >= test_auc:
        print("  Using selected feature model (better or equal performance)")
        model = model_selected
        y_pred_test = y_pred_test_selected
        y_pred_proba_test = y_pred_proba_test_selected
        test_accuracy = test_accuracy_selected
        test_auc = test_auc_selected
        feature_columns = selected_features

        # Update feature importance
        feature_importance = pd.DataFrame({
            'feature': feature_columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
    else:
        print("  Keeping original model (better performance)")

# Analyze time decay feature importance
time_decay_features = ['views_exp_decay_30', 'likes_exp_decay_30',
                       'views_recency_weighted', 'likes_recency_weighted',
                       'views_per_day', 'likes_per_day']
decay_importance = feature_importance[feature_importance['feature'].isin(time_decay_features)]
total_decay_importance = decay_importance['importance'].sum()

original_engagement = ['view_count', 'like_count', 'comment_count']
original_importance = feature_importance[feature_importance['feature'].isin(original_engagement)]
total_original_importance = original_importance['importance'].sum()

print(f"\nTime decay importance: {total_decay_importance*100:.2f}%")
print(f"Raw engagement importance: {total_original_importance*100:.2f}%")

# Save results
feature_importance.to_csv('feature_importance.csv', index=False)

# Visualizations
print("\nGenerating visualizations...")

# Plot 1: Top 20 Features
plt.figure(figsize=(10, 8))
top_features = feature_importance.head(20)
colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(top_features)))
bars = plt.barh(range(len(top_features)), top_features['importance'], color=colors)
plt.yticks(range(len(top_features)), top_features['feature'], fontsize=10)
plt.xlabel('Importance', fontsize=12)
plt.title('Top 20 Most Important Features', fontsize=14, fontweight='bold')
plt.gca().invert_yaxis()
plt.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.savefig(os.path.join(output_folder, 'top_features.png'), dpi=300, bbox_inches='tight')
plt.close()

# Plot 2: Confusion Matrix
cm = confusion_matrix(y_test, y_pred_test)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True)
plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
plt.ylabel('Actual', fontsize=12)
plt.xlabel('Predicted', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(output_folder, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
plt.close()

# Plot 3: Feature Categories
feature_categories = {
    'Trailer': ['has_trailer_data', 'view_count', 'like_count', 'comment_count', 'like_rate', 'comment_rate'],
    'Time Decay': time_decay_features,
    'Production': ['budget', 'budget_log', 'has_budget', 'runtime', 'production_company_count', 'production_country_count'],
    'Budget Bins': [f'budget_{tier}' for tier in ['micro', 'low', 'medium', 'high', 'blockbuster']],
    'Runtime Bins': [f'runtime_{cat}' for cat in ['short', 'standard', 'long', 'epic', 'marathon']],
    'Engagement Bins': [f'engagement_{tier}' for tier in ['none', 'very_low', 'low', 'medium', 'high', 'viral']],
    'Temporal': ['release_year', 'release_month', 'release_quarter', 'is_summer_release', 'is_holiday_release'] + decade_cols,
    'People': ['cast_count', 'director_count', 'has_cast', 'has_director'],
    'Genre': ['genre_count', 'has_genre'] + [f'is_{g.lower().replace(" ", "_")}' for g in top_genres],
    'Sentiment': ['has_sentiment', 'sentiment_positive', 'sentiment_negative', 'sentiment_magnitude',
                  'sentiment_strength', 'very_positive', 'very_negative', 'neutral_sentiment',
                  'moderate_positive', 'moderate_negative'],
    'Production Companies': company_cols,
    'External': ['has_rt_url', 'has_critic_score', 'has_audience_score']
}

category_importance = {}
for category, features in feature_categories.items():
    cat_features = [f for f in features if f in feature_importance['feature'].values]
    importance_sum = feature_importance[feature_importance['feature'].isin(cat_features)]['importance'].sum()
    category_importance[category] = importance_sum

plt.figure(figsize=(10, 6))
categories = list(category_importance.keys())
importances = list(category_importance.values())
colors = plt.cm.Set3(np.linspace(0, 1, len(categories)))
bars = plt.bar(categories, importances, color=colors, edgecolor='black', linewidth=1.5)
plt.ylabel('Total Importance', fontsize=12)
plt.title('Feature Importance by Category', fontsize=14, fontweight='bold')
plt.xticks(rotation=45, ha='right')
plt.grid(True, alpha=0.3, axis='y')
for bar, val in zip(bars, importances):
    plt.text(bar.get_x() + bar.get_width()/2, val, f'{val:.3f}',
             ha='center', va='bottom', fontsize=10, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(output_folder, 'category_importance.png'), dpi=300, bbox_inches='tight')
plt.close()

# Plot 4: ROC Curve
from sklearn.metrics import roc_curve

fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba_test)
plt.figure(figsize=(8, 8))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {test_auc:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random classifier')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=14, fontweight='bold')
plt.legend(loc="lower right", fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(output_folder, 'roc_curve.png'), dpi=300, bbox_inches='tight')
plt.close()

# Plot 5: Prediction Distribution
plt.figure(figsize=(10, 6))
plt.hist(y_pred_proba_test[y_test == 0], bins=50, alpha=0.6, label='Unsuccessful (Actual)', color='red', edgecolor='black')
plt.hist(y_pred_proba_test[y_test == 1], bins=50, alpha=0.6, label='Successful (Actual)', color='green', edgecolor='black')
plt.xlabel('Predicted Probability of Success', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.title('Distribution of Predicted Probabilities', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(os.path.join(output_folder, 'prediction_distribution.png'), dpi=300, bbox_inches='tight')
plt.close()

# Plot 6: Performance by Release Year
test_df = df_model.loc[X_test.index].copy()
test_df['prediction'] = y_pred_test
test_df['actual'] = y_test.values

yearly_performance = test_df.groupby('release_year').agg({
    'actual': 'mean',
    'prediction': 'mean'
}).reset_index()

plt.figure(figsize=(12, 6))
plt.plot(yearly_performance['release_year'], yearly_performance['actual'],
         marker='o', linewidth=2, markersize=8, label='Actual Success Rate', color='blue')
plt.plot(yearly_performance['release_year'], yearly_performance['prediction'],
         marker='s', linewidth=2, markersize=8, label='Predicted Success Rate', color='orange')
plt.xlabel('Release Year', fontsize=12)
plt.ylabel('Success Rate', fontsize=12)
plt.title('Model Performance by Release Year (Test Set)', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(output_folder, 'performance_by_year.png'), dpi=300, bbox_inches='tight')
plt.close()

# Plot 7: Feature Distribution (Top Features)
top_5_features = feature_importance.head(5)['feature'].tolist()
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for idx, feature in enumerate(top_5_features):
    if feature in df_model.columns:
        ax = axes[idx]
        successful = df_model[df_model['is_successful'] == 1][feature]
        unsuccessful = df_model[df_model['is_successful'] == 0][feature]

        ax.hist(unsuccessful, bins=30, alpha=0.6, label='Unsuccessful', color='red', edgecolor='black')
        ax.hist(successful, bins=30, alpha=0.6, label='Successful', color='green', edgecolor='black')
        ax.set_xlabel(feature, fontsize=10)
        ax.set_ylabel('Frequency', fontsize=10)
        ax.set_title(f'{feature}', fontsize=11, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')

# Remove extra subplot
axes[-1].axis('off')

plt.suptitle('Distribution of Top 5 Features by Success', fontsize=14, fontweight='bold', y=1.00)
plt.tight_layout()
plt.savefig(os.path.join(output_folder, 'feature_distributions.png'), dpi=300, bbox_inches='tight')
plt.close()

# Plot 11: Budget vs Success Rate
if 'budget' in df_model.columns:
    budget_bins = pd.qcut(df_model[df_model['budget'] > 0]['budget'], q=10, duplicates='drop')
    budget_success = df_model[df_model['budget'] > 0].groupby(budget_bins)['is_successful'].mean()

    plt.figure(figsize=(12, 6))
    x_pos = range(len(budget_success))
    plt.bar(x_pos, budget_success.values, color='steelblue', edgecolor='black', linewidth=1.5)
    plt.xlabel('Budget Range', fontsize=12)
    plt.ylabel('Success Rate', fontsize=12)
    plt.title('Success Rate by Budget Range', fontsize=14, fontweight='bold')
    plt.xticks(x_pos, [f'${int(interval.left/1e6)}-{int(interval.right/1e6)}M' for interval in budget_success.index],
               rotation=45, ha='right', fontsize=9)
    plt.grid(True, alpha=0.3, axis='y')
    plt.axhline(y=df_model['is_successful'].mean(), color='red', linestyle='--',
                linewidth=2, label=f'Overall avg: {df_model["is_successful"].mean():.2%}')
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'budget_vs_success.png'), dpi=300, bbox_inches='tight')
    plt.close()

