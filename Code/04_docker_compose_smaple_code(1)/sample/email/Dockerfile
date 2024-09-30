# Base image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy all files to the container
COPY . /app

# Install dependencies
RUN pip install flask scikit-learn

# Expose the port
EXPOSE 5001

# Command to run the app
CMD ["python", "email.py"]
