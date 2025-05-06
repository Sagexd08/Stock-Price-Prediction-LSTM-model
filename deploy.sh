#!/bin/bash

echo "Building and deploying Stock Price Prediction app to Firebase..."

# Create public directory if it doesn't exist
mkdir -p public

# Copy static assets to public directory
echo "Copying static assets..."
mkdir -p public/assets
cp -r assets/* public/assets/ 2>/dev/null || :

# Install Firebase CLI if not already installed
if ! command -v firebase &> /dev/null; then
    echo "Installing Firebase CLI..."
    npm install -g firebase-tools
fi

# Login to Firebase (if not already logged in)
firebase login

# Initialize the project (if not already initialized)
if [ ! -f .firebaserc ]; then
    echo "Initializing Firebase project..."
    firebase init hosting functions
fi

# Install dependencies for Firebase Functions
echo "Installing dependencies for Firebase Functions..."
cd functions
npm install
cd ..

# Build Docker image
echo "Building Docker image..."
docker build -t stock-prediction-firebase .

# Deploy to Firebase
echo "Deploying to Firebase..."
firebase deploy

echo "Deployment completed!"

# Open the deployed app in the browser
echo "Opening the deployed app in the browser..."
if [[ "$OSTYPE" == "darwin"* ]]; then
    open https://stock-price-prediction-app.web.app
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    xdg-open https://stock-price-prediction-app.web.app
else
    echo "Please open https://stock-price-prediction-app.web.app in your browser"
fi
