# Import necessary libraries
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Step 1: Sample email data (you should have more data for better results)
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

# Step 2: Vectorization and classification
# Create a pipeline with a CountVectorizer and a Naive Bayes model
model = make_pipeline(CountVectorizer(lowercase=True, stop_words='english'), MultinomialNB())

# Step 3: Train the model
model.fit(emails, labels)

# Step 4: Test with new emails
new_emails = [
    "Claim your free iPhone now by clicking this link!",
    "Your Amazon package has been delivered, thank you for your purchase.",
    "Win a free car! Enter our giveaway contest.",
    "Your work report is due by the end of the week. Please submit it on time."
]

# Predict the labels for new emails
predictions = model.predict(new_emails)

# Step 5: Display results
for email, label in zip(new_emails, predictions):
    print(f"Email: '{email}' is classified as: {'Spam' if label == 1 else 'Not Spam'}")

# Optional: Evaluate the accuracy with a train/test split
X_train, X_test, y_train, y_test = train_test_split(emails, labels, test_size=0.2, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
