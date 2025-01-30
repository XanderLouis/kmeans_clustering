
# Customer Segmentation with K-Means Clustering

## Overview

This project is a **K-Means Clustering app** built with **Streamlit and FastAPI**. It segments customers based on their **annual income** and **spending score**, helping businesses identify different spending behaviors.

## Features

- **Interactive Web App:** Users input their annual income and spending score to predict their segment.
- **Machine Learning Model:** Utilizes **K-Means Clustering** for segmentation.
- **FastAPI Backend:** Handles model inference for predictions.
- **Streamlit Frontend:** Provides an easy-to-use interface.
- **Visualization:** Displays customer clusters and centroids.

## Installation & Setup

### Prerequisites

Ensure you have **Python 3.7+** installed.

### Clone the Repository

```bash
git clone https://github.com/yourusername/kmeans-streamlit-app.git
cd kmeans-streamlit-app
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run the FastAPI Backend

```bash
uvicorn api:app --host 0.0.0.0 --port 8000
```

### Run the Streamlit App

```bash
streamlit run app.py
```

## Usage

1. Enter the **Annual Income (k\$)** and **Spending Score (1-100)**.
2. Click **Predict Customer's Segmentation**.
3. The app will display the predicted segment:
   - **Prudent Spender**
   - **Generous Spender**
   - **Extravagant Spender**
   - **Wise Spender**
   - **Loose Spender**
4. A visualization of customer clusters is also available.

## Project Structure

```
📂 kmeans-streamlit-app
├── 📂 model
│   ├── model.pkl  # Trained K-Means model
├── 📂 data
│   ├── data_clustering.csv  # Customer dataset
├── 📂 src
│   ├── Figure_2.png  # Cluster visualization
├── app.py  # Streamlit frontend
├── api.py  # FastAPI backend
├── train.py  # Model training script
├── requirements.txt  # Dependencies
└── README.md  # Project documentation
```

## Technologies Used

- **Streamlit** – For the web interface.
- **FastAPI** – For handling API requests.
- **Scikit-learn** – For K-Means clustering.
- **Joblib** – For model serialization.
- **Pandas & NumPy** – For data handling.
- **Matplotlib** – For visualization.

## Contributing

Pull requests are welcome! Feel free to suggest further improvements.


