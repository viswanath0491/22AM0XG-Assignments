from flask import Flask, request, jsonify
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

app = Flask(__name__)

# Movie reviews data
reviews = [
    "This movie was fantastic, I loved it!", 
    "The plot was boring and predictable", 
    "Amazing performance by the cast", 
    "I wasted two hours of my life", 
    "Brilliant direction and storytelling", 
    "The movie was a disaster", 
    "What a great film!", 
    "It was an awful experience",
    "The cinematography was beautiful", 
    "Terrible acting and poor script"
]
labels = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]  # 1: Positive, 0: Negative

# Vectorizer and classifier
model = make_pipeline(CountVectorizer(lowercase=True, stop_words='english'), MultinomialNB())
model.fit(reviews, labels)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    new_reviews = data.get('reviews', [])
    predictions = model.predict(new_reviews)
    results = [{'review': review, 'label': 'Positive' if label == 1 else 'Negative'} for review, label in zip(new_reviews, predictions)]
    return jsonify(results)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
