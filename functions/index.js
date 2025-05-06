const functions = require("firebase-functions");
const admin = require("firebase-admin");
const { exec } = require("child_process");
const os = require("os");
const path = require("path");
const fs = require("fs");
const express = require("express");
const cors = require("cors");
const { createProxyMiddleware } = require("http-proxy-middleware");

admin.initializeApp();

// Create an Express app for the Streamlit proxy
const app = express();

// Enable CORS
app.use(cors({ origin: true }));

// Cloud Function to serve the Streamlit app
exports.streamlitApp = functions
  .runWith({
    memory: "2GB",
    timeoutSeconds: 540,
    cpu: 2,
  })
  .https.onRequest(async (req, res) => {
    // In a production environment, you would:
    // 1. Run Streamlit in a container (e.g., Cloud Run)
    // 2. Proxy requests to that container
    // 3. Handle authentication and other middleware here

    // For demo purposes, we'll redirect to a hosted version
    // In a real implementation, you would use a proxy to the Streamlit server

    // Check if the user is authenticated
    const idToken = req.headers.authorization?.split("Bearer ")[1];

    if (idToken) {
      try {
        // Verify the ID token
        await admin.auth().verifyIdToken(idToken);

        // User is authenticated, redirect to the app
        // In production, this would proxy to your Streamlit container
        res.redirect("https://stock-price-prediction-app.web.app");
      } catch (error) {
        console.error("Error verifying ID token:", error);
        res.status(401).json({ error: "Unauthorized" });
      }
    } else {
      // No token provided, redirect to login page
      res.redirect("https://stock-price-prediction-app.web.app/login");
    }
  });

// Cloud Function to update stock data
exports.updateStockData = functions.pubsub
  .schedule("0 0 * * *") // Run daily at midnight
  .timeZone("America/New_York")
  .onRun(async (context) => {
    // This function would download fresh stock data
    // In a real implementation, you would use the yfinance API
    // and store the data in Firebase Storage or Firestore

    console.log("Updating stock data...");

    // Example code to download data (not actually executed)
    /*
    const tempDir = os.tmpdir();
    const scriptPath = path.join(tempDir, "download_data.py");

    // Write a Python script to download data
    fs.writeFileSync(
      scriptPath,
      `
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

# Download data for popular stocks
tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]
end_date = datetime.now()
start_date = end_date - timedelta(days=365)

for ticker in tickers:
    data = yf.download(ticker, start=start_date, end=end_date)
    data.to_csv(f"{ticker}_data.csv")
    print(f"Downloaded data for {ticker}")
      `
    );

    // Execute the script
    return new Promise((resolve, reject) => {
      exec(`python ${scriptPath}`, (error, stdout, stderr) => {
        if (error) {
          console.error(`Error: ${error}`);
          reject(error);
          return;
        }
        console.log(`stdout: ${stdout}`);
        console.error(`stderr: ${stderr}`);
        resolve();
      });
    });
    */

    console.log("Stock data update completed");
    return null;
  });
