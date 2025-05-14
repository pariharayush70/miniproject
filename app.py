from flask import Flask, request, render_template
import pickle
import pandas as pd

app = Flask(__name__)

# Load the trained model and movie mappings
with open('svd_model.pkl', 'rb') as f:
    svd = pickle.load(f)
with open('movie_id_to_title.pkl', 'rb') as f:
    movie_id_to_title = pickle.load(f)

# Load movie IDs from ratings for prediction
ratings = pd.read_csv('ratings_small.csv')
movie_ids = ratings['movieId'].unique()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    user_id = int(request.form['user_id'])
    n_recommendations = int(request.form.get('n_recommendations', 5))

    # Predict ratings for all movies for the given user
    predictions = []
    for movie_id in movie_ids:
        pred = svd.predict(user_id, movie_id)
        predictions.append((movie_id, pred.est))

    # Sort predictions by estimated rating and get top N
    predictions.sort(key=lambda x: x[1], reverse=True)
    top_predictions = predictions[:n_recommendations]

    # Map movie IDs to titles
    recommendations = []
    for movie_id, score in top_predictions:
        title = movie_id_to_title.get(movie_id, f"Movie ID {movie_id}")
        recommendations.append((title, round(score, 2)))

    return render_template('index.html', recommendations=recommendations, user_id=user_id)

if __name__ == '__main__':
    app.run(debug=True)