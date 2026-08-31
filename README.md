# 🏏 IPL Match Predictor

A machine learning-powered web application that predicts the outcome and scores of Indian Premier League (IPL) matches in real-time. Built with Streamlit, this tool leverages historical delivery and match data to provide statistical insights during a game.

## ✨ Features

* First Innings Projection (Regression): Predicts the total runs the batting team will score by the end of the innings based on current runs, wickets, and overs.
* Second Innings Win Probability (Classification): Calculates the real-time probability of the chasing team winning the match, dynamically updating based on the target score and match situation.
* Interactive UI: Clean, intuitive interface for seamless data entry and instant visual feedback (progress bars, situational alerts).
* Intelligent Input Handling: Converts standard cricket over notation (e.g., 7.2 for 7 overs and 2 balls) into decimal formats readable by the ML models.

## 🛠️ Tech Stack

* Frontend: Streamlit
* Backend/Data Processing: Python, NumPy
* Machine Learning: Scikit-Learn (models saved via joblib)

## 📁 Project Structure

├── app.py                   # Main Streamlit application script
├── ipl_win_predictor.pkl    # Pre-trained classification model (Win Probability)
├── ipl_runs_predictor.pkl   # Pre-trained regression model (Total Runs)
├── le_batting.pkl           # Label encoder for batting teams
├── le_bowling.pkl           # Label encoder for bowling teams
├── requirements.txt         # Project dependencies (Create this file)
└── README.txt               # Project documentation
