@echo off
echo Building and deploying Stock Price Prediction app to Firebase...

REM Create public directory if it doesn't exist
if not exist public mkdir public

REM Copy static assets to public directory
echo Copying static assets...
if not exist public\assets mkdir public\assets
xcopy /E /Y assets public\assets

REM Install Firebase CLI if not already installed
where firebase >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo Installing Firebase CLI...
    npm install -g firebase-tools
)

REM Login to Firebase (if not already logged in)
firebase login

REM Initialize the project (if not already initialized)
if not exist .firebaserc (
    echo Initializing Firebase project...
    firebase init hosting functions
)

REM Install dependencies for Firebase Functions
echo Installing dependencies for Firebase Functions...
cd functions
npm install
cd ..

REM Build Docker image
echo Building Docker image...
docker build -t stock-prediction-firebase .

REM Deploy to Firebase
echo Deploying to Firebase...
firebase deploy

echo Deployment completed!

REM Open the deployed app in the browser
echo Opening the deployed app in the browser...
start https://stock-price-prediction-app.web.app
