https://luxeestimate.streamlit.app/

🏡 LuxeEstimate Pro-Smart-House-Price-Evaluation
AI-Powered Property Valuation & Insights with Real-Time Market Intelligence

LuxeEstimate Pro is an advanced property valuation system built with Streamlit and Machine Learning. It predicts house prices, provides neighborhood-level insights, integrates real-time APIs for crime, amenities, and weather, and offers an interactive AI-powered property advisor chatbot.

🚀 Features

- 🔮 AI-Powered Valuation – Predicts property prices using a *Random Forest Regressor* trained on enriched neighborhood data.  
- 📊 Neighborhood Insights Dashboard – Explore price trends, compare localities, and analyze investment potential.  
- 🌤️ Real-Time Weather Integration – Uses OpenWeather API to display live city weather alongside property valuations.  
- 📰 Crime Data Intelligence – Fetches recent crime-related news via **NewsAPI** to score locality safety.  
- 🏬 Amenity Scoring – Uses **OpenStreetMap + Overpass API** to evaluate nearby facilities (schools, hospitals, markets, parks, etc.).  
- 💬 AI Chatbot Property Advisor – Interactive assistant that answers property queries with data-driven insights.  
- 📈 Investment Potential Analytics – ROI scoring, neighborhood rankings, and price-per-sqft projections.  
- 🎨 Modern UI – Styled with custom CSS, Plotly charts, and interactive Streamlit widgets.  

---

🏗️ Tech Stack

 Core
- Python 3.9+  
- Streamlit – UI & app framework  
- Pandas, NumPy – Data processing  
- Scikit-Learn – ML model (Random Forest Regressor)  
- Plotly Express – Interactive data visualization  
- Folium / streamlit-folium – Geospatial insights  

 APIs & Integrations
- OpenWeather API – Live weather data  
- NewsAPI – Crime and locality-related news scoring  
- Nominatim + Overpass API (OSM) – Amenity-based neighborhood scoring  
- OpenAI API (optional) – Conversational enhancements  

 Supporting Tools
- Excel/CSV datasets – Property transaction data  
- Jupyter Notebook (`weather_home_prediction_.ipynb`) – Data cleaning, preprocessing, and experiments  
- Streamlit Secrets – Secure storage of API keys

 Data Cleaning & Preprocessing

From `weather_home_prediction_.ipynb` and integrated functions:

- Standardized column names (lowercase, stripped spaces).  
- Converted `price` and `size` to numeric.  
- Derived new features:  
  - `price_per_sqft = price / size_sqft`  
  - `neighborhood_rank` = relative ranking within each city  
- Filled missing values for `beds`, `baths`, and `type`.  
- Normalized city and neighborhood names (Title case).  
- Ensured date column is valid (`today` if missing).  

---

⚡ REST APIs Used

1. 🌤️ OpenWeather API – Retrieves live weather for valuation context.  
2. 📰 NewsAPI – Fetches recent crime-related articles → converts into a crime safety score.  
3. 🗺️ Nominatim + Overpass API – Geocodes neighborhoods & fetches nearby amenities → converts into an amenity score.  
4. 🤖 OpenAI API (Optional) – Enhances chatbot responses.  

---

 📊 Machine Learning Model

- Algorithm: Random Forest Regressor (`n_estimators=200`, `max_depth=10`)  
- Features:  
  - Numerical: `beds`, `baths`, `size_sqft`, `neighborhood_rank`  
  - Categorical: `city`, `type`, `neighborhood`  
- Target: `price`  
- Metrics:  
  - Train R²: ~0.9  
  - Test R²: ~0.8  
  - MAE: ~₹X0,000 (varies by dataset)  

---

## 🔧 Installation

```bash
# Clone repository
git clone https://github.com/<your-username>/luxeestimate-pro.git
cd luxeestimate-pro

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate   # On Windows
source .venv/bin/activate  # On Mac/Linux

# Install dependencies
pip install -r requirements.txt

# Run app
streamlit run luxeestimate_pro.py
```
Documentation & Insights

-Prediction pipeline: Explore Predict.py for the model logic.
-Data handling: View how you cleaned and analyzed your data in the notebook weather_home_prediction_.ipynb.

🔑 API Keys Setup
openweather_key = "your_openweather_api_key"
newsapi_key = "your_newsapi_key"
openai_key = "your_openai_key"

<img width="1365" height="620" alt="image" src="https://github.com/user-attachments/assets/2ae40d02-4fb9-4859-a20e-a3e882fa66e6" />
<img width="1365" height="509" alt="image" src="https://github.com/user-attachments/assets/e6d3d9a0-123a-4331-a774-68db7a16b780" />
<img width="1365" height="634" alt="image" src="https://github.com/user-attachments/assets/609130a1-8046-4b9d-afff-fcaa8f761b72" />
<img width="1361" height="634" alt="image" src="https://github.com/user-attachments/assets/31ba7ccc-177b-4319-bad0-cc3e949746c0" />

📜 License
This project is licensed under the MIT License.
MIT License

Copyright (c) 2025 Ritika C. Bobhate

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
...

✨ Future Enhancements :

-Integrate rental yield predictions
-Add sentiment analysis on news articles
-Support for multiple countries (LANDSAT SENITAL REAL TIME DATA)
-Integration with property listing APIs

👩‍💻 Author
Ritika C. Bobhate
Built as part of learning AI, ML, and real estate analytics.

  
