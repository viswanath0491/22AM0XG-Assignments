from flask import Flask, request, jsonify
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

app = Flask(__name__)

# Sample email data
emails = [
    "Congratulations! You've won a $1000 gift card, click here to claim now",
    "Dear customer, your invoice for the month of August is attached. Please pay it by the end of the month.",
    "Exclusive offer! Get 50% off on your next purchase, hurry up before the offer ends!",
    "Your bank statement is available for viewing online. Please log in to your account to review.",
    "Act now! You have a chance to win a free trip to Paris. Click the link to participate.",
    "Your meeting has been scheduled for 2 PM tomorrow. Please confirm your attendance.",
    "Urgent! Your account has been suspended due to suspicious activity. Verify your account to restore access.",
    "Reminder: The project report is due tomorrow. Please submit it by the end of the day.",
    "Your order has been shipped and is expected to arrive within 3 business days.",
    "You have a new voicemail from your phone service provider."
]
labels = [1, 0, 1, 0, 1, 0, 1, 0, 0, 0]  # 1: Spam, 0: Not Spam

# Vectorizer and classifier
model = make_pipeline(CountVectorizer(lowercase=True, stop_words='english'), MultinomialNB())
model.fit(emails, labels)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    new_emails = data.get('emails', [])
    predictions = model.predict(new_emails)
    results = [{'email': email, 'label': 'Spam' if label == 1 else 'Not Spam'} for email, label in zip(new_emails, predictions)]
    return jsonify(results)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5001)
