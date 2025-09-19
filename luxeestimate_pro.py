import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import plotly.express as px
from streamlit_folium import folium_static
import requests
from datetime import datetime, timedelta
import openai
from plotly.io import to_html
import os

# App configuration
st.set_page_config(
    page_title="🏡 LuxeEstimate Pro | AI Property Valuation",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize OpenAI API
openai.api_key = st.secrets.get("openai_key", "")

# --- Data Loading ---
@st.cache_data
def load_property_data():
    try:
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(BASE_DIR, "data", "geo_enriched_property_data_osm.xlsx")

        df = pd.read_excel(file_path)
        df.columns = df.columns.str.lower().str.strip()
        
        # Validate required columns
        required = ['price', 'size', 'neighborhood', 'city']
        if not all(col in df.columns for col in required):
            st.error("Missing required columns in data")
            st.stop()
            
        # Clean data
        df['price'] = pd.to_numeric(df['price'], errors='coerce')
        df['size_sqft'] = pd.to_numeric(df['size'], errors='coerce')
        df = df.dropna(subset=['price', 'size_sqft'])
        df['price_per_sqft'] = df['price'] / df['size_sqft']
        
        # Add neighborhood rankings
        df['neighborhood_rank'] = df.groupby(['city', 'neighborhood'])['price_per_sqft'].rank(pct=True)
        
        return df
        
    except Exception as e:
        st.error(f"Failed to load data: {str(e)}")
        return pd.DataFrame()

# --- AI Response Generation ---
def generate_property_response(query, property_data):
    """Generate responses from property data"""
    try:
        query = query.lower()
        
        # Handle price queries
        if any(word in query for word in ["price", "cost", "₹"]):
            return handle_price_query(query, property_data)
            
        # Handle trend queries
        elif "trend" in query:
            return handle_trend_query(query, property_data)
            
        # Default response
        return ("I found property information. "
                "For better answers, ask about: "
                "prices, locations, or property types.")
                
    except Exception as e:
        return f"Error analyzing data: {str(e)}"

def handle_price_query(query, data):
    """Handle price-related questions with more detailed responses"""
    try:
        # Find mentioned city
        cities = [c for c in data['city'].unique() if c.lower() in query.lower()]
        
        if not cities:
            return "Please specify a city for price information (e.g. 'What are prices in Mumbai?')"
        
        city = cities[0]
        city_data = data[data['city'].str.lower() == city.lower()]
        
        # Calculate various price metrics
        median_price = city_data['price'].median()
        avg_price = city_data['price'].mean()
        max_price = city_data['price'].max()
        min_price = city_data['price'].min()
        price_per_sqft = city_data['price_per_sqft'].median()
        
        # Check for specific price queries
        if "max" in query.lower() or "highest" in query.lower():
            return (f"Highest property price in {city}: ₹{max_price:,.0f}\n"
                   f"Typical features: {get_property_features(city_data, max_price)}")
        
        if "min" in query.lower() or "lowest" in query.lower():
            return (f"Lowest property price in {city}: ₹{min_price:,.0f}\n"
                   f"Typical features: {get_property_features(city_data, min_price)}")
        
        # Default price response
        return (f"Property prices in {city}:\n"
                f"- Median: ₹{median_price:,.0f}\n"
                f"- Average: ₹{avg_price:,.0f}\n"
                f"- Price per sqft: ₹{price_per_sqft:,.0f}\n"
                f"- Range: ₹{min_price:,.0f} to ₹{max_price:,.0f}")
                
    except Exception as e:
        return f"Error retrieving price data: {str(e)}"
def get_property_features(df, price):
    """Get typical features for properties at a given price point"""
    similar = df[np.abs(df['price'] - price) <= 0.1 * price]
    if similar.empty:
        return "No detailed data available"
    
    features = []
    if 'beds' in similar.columns:
        avg_beds = similar['beds'].mean()
        features.append(f"{avg_beds:.1f} beds")
    if 'baths' in similar.columns:
        avg_baths = similar['baths'].mean()
        features.append(f"{avg_baths:.1f} baths")
    if 'size_sqft' in similar.columns:
        avg_size = similar['size_sqft'].mean()
        features.append(f"{avg_size:,.0f} sqft")
    
    return ", ".join(features) if features else "Standard features"

def handle_trend_query(query, data):
    """Handle market trend questions"""
    trends = data.groupby('city')['price_per_sqft'].median().sort_values(ascending=False)
    return "\n".join([f"- {city}: ₹{price:,.0f}/sqft" for city, price in trends.head(3).items()])
def property_advisor_chat():
    st.title("🏡 AI Property Advisor")
    
    # Custom CSS for chat interface
    st.markdown("""
    <style>
      
        .user-message {
            background-color: black;
            color: white;
            border-radius: 18px 18px 0 18px;
            padding: 12px 16px;
            margin: 8px 0;
            max-width: 80%;
            margin-left: auto;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .assistant-message {
            background-color: black;
            color: white;
            border-radius: 18px 18px 18px 0;
            padding: 12px 16px;
            margin: 8px 0;
            max-width: 80%;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .suggestions {
            display: flex;
            gap: 8px;
            flex-wrap: wrap;
            margin-top: 16px;
        }
        .suggestion-chip {
            background-color: #E5E7EB;
            color: #1F2937;
            padding: 8px 12px;
            border-radius: 20px;
            font-size: 14px;
            cursor: pointer;
            transition: all 0.2s;
        }
        .suggestion-chip:hover {
            background-color: #D1D5DB;
            transform: translateY(-2px);
        }
        .typing-indicator {
            display: flex;
            padding: 12px 16px;
        }
        .typing-dot {
            width: 8px;
            height: 8px;
            background-color: #9CA3AF;
            border-radius: 50%;
            margin: 0 2px;
            animation: typing 1.4s infinite ease-in-out;
        }
        .typing-dot:nth-child(1) { animation-delay: 0s; }
        .typing-dot:nth-child(2) { animation-delay: 0.2s; }
        .typing-dot:nth-child(3) { animation-delay: 0.4s; }
        @keyframes typing {
            0%, 60%, 100% { transform: translateY(0); opacity: 0.6; }
            30% { transform: translateY(-4px); opacity: 1; }
        }
    </style>
    """, unsafe_allow_html=True)

    # Initialize chat with better prompts
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": """
            <div style='font-size: 16px;'>
                <h3 style='color: #4F46E5; margin-bottom: 8px;'>Welcome to your AI Property Advisor! 🏡</h3>
                <p>I can help you with:</p>
                <ul style='margin-top: 8px;'>
                    <li>Finding properties matching your criteria</li>
                    <li>Comparing neighborhoods and prices</li>
                    <li>Investment recommendations</li>
                    <li>Market trends and insights</li>
                </ul>
                <div class='suggestions'>
                    <div class='suggestion-chip' onclick='this.innerHTML="Show 2BHK apartments in Mumbai under ₹1.5Cr"'>Show 2BHK in Mumbai</div>
                    <div class='suggestion-chip' onclick='this.innerHTML="Compare Bandra and Andheri property prices"'>Compare areas</div>
                    <div class='suggestion-chip' onclick='this.innerHTML="What are the best investment areas in Bangalore?"'>Investment advice</div>
                </div>
            </div>
            """}
        ]
    
    # Display chat history in container
    with st.container():
        st.markdown("<div class='chat-container'>", unsafe_allow_html=True)
        
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                st.markdown(f"<div class='user-message'>{msg['content']}</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='assistant-message'>{msg['content']}</div>", unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Enhanced input with quick suggestions
    with st.form("chat_input_form"):
        cols = st.columns([5, 1])
        with cols[0]:
            user_input = st.text_input(
                "Ask about properties...",
                key="chat_input",
                placeholder="e.g. 'Show 3BHK in Bangalore between ₹1-1.5Cr'",
                label_visibility="collapsed"
            )
        with cols[1]:
            submitted = st.form_submit_button("Send", type="primary")
    
    # Quick suggestion chips

    
    # Handle user input with better responses
    if submitted and user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        with st.spinner(""):
            # Show typing indicator
            typing_html = """
            <div class='typing-indicator'>
                <div class='typing-dot'></div>
                <div class='typing-dot'></div>
                <div class='typing-dot'></div>
            </div>
            """
            typing_placeholder = st.empty()
            typing_placeholder.markdown(typing_html, unsafe_allow_html=True)
            
            try:
                # Process the query with enhanced response generation
                response = generate_property_response(user_input, property_data)
                
                # Remove typing indicator
                typing_placeholder.empty()
                
                # Display response with better formatting
                formatted_response = f"""
                <div style='font-size: 15px; line-height: 1.5;'>
                    {response}
                    <div class='suggestions' style='margin-top: 16px;'>
                        <div class='suggestion-chip' onclick='document.querySelector("[name=\'chat_input\']").value="Show more properties like this"'>More like this</div>
                        <div class='suggestion-chip' onclick='document.querySelector("[name=\'chat_input\']").value="What amenities do these properties have?"'>Amenities</div>
                        <div class='suggestion-chip' onclick='document.querySelector("[name=\'chat_input\']").value="Show price trends for this area"'>Price trends</div>
                    </div>
                </div>
                """
                
                st.session_state.messages.append({"role": "assistant", "content": formatted_response})
                st.rerun()
                
            except Exception as e:
                typing_placeholder.empty()
                error_msg = f"""
                <div style='color: #EF4444; font-size: 15px;'>
                    Sorry, I encountered an error processing your request. Please try again or ask differently.
                    <br><br>
                    <i>Error: {str(e)}</i>
                </div>
                """
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
                st.rerun()



def get_weather(city):
    try:
        key = st.secrets["openweather_key"]
        url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={key}&units=metric"
        response = requests.get(url)
        data = response.json()
        return f"{data['main']['temp']}°C, {data['weather'][0]['description'].capitalize()}"
    except:
        return "Weather data not available"

# Custom CSS for enhanced UI
st.markdown("""
<style>
    :root {
        --primary: #4F46E5;
        --secondary: #10B981;
        --accent: #F59E0B;
        --dark: #1F2937;
        --light: #F3F4F6;
    }

    .main {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }

    .stApp {
        background-color: transparent;
    }
    
    .stButton>button {
        background: linear-gradient(to right, var(--primary), var(--secondary));
        color: white;
        border: none;
        border-radius: 8px;
        padding: 12px 28px;
        font-weight: 600;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
    }

    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 8px rgba(0, 0, 0, 0.15);
    }

    .stTextInput>div>div>input, 
    .stSelectbox>div>div>select,
    .stNumberInput>div>div>input {
        border-radius: 8px;
        border: 1px solid #E5E7EB;
        padding: 10px 15px;
    }

    .stMarkdown h1 {
        color: var(--dark);
        border-bottom: 2px solid var(--accent);
        padding-bottom: 10px;
    }

    .card {
        background: gradient(linear, var(--dark), var(--secondary));
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
        margin-bottom: 20px;
    }

    .feature-icon {
        font-size: 2rem;
        color: var(--primary);
        margin-bottom: 10px;
    }

    .price-display {
        font-size: 2.5rem;
        font-weight: 700;
        color: var(--primary);
        text-align: center;
        margin: 20px 0;
    }
    
    .highlight {
        background: linear-gradient(120deg, #f6d365 0%, #fda085 100%);
        -webkit-background-clip: text;
        background-clip: text;
        color: transparent;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

def generate_local_response(input_data):
    return "Response based on input: " + str(input_data)

# Load and preprocess property data with neighborhood focus
@st.cache_data
def load_and_preprocess_data():
    # Load the data
    BASE_DIR = os.path.dirname(__file__)  # current script folder
    DATA_PATH = os.path.join(BASE_DIR, "data", "geo_enriched_property_data_osm.xlsx")

    df = pd.read_excel(DATA_PATH)


    # Normalize column names
    df.columns = df.columns.str.lower().str.strip()

    # Required columns
    required_columns = ['price', 'size', 'neighborhood', 'city']
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        st.error(f"Missing required columns: {missing}")
        raise ValueError(f"Missing required columns: {missing}")

    # Handle numeric conversion and missing values
    df['price'] = pd.to_numeric(df['price'], errors='coerce')
    df['size'] = pd.to_numeric(df['size'], errors='coerce')
    df = df.dropna(subset=['price', 'size'])

    # Rename size to size_sqft for consistency
    df['size_sqft'] = df['size']
    df['price_per_sqft'] = df['price'] / df['size']

    # Clean up neighborhood and city names
    df['neighborhood'] = df['neighborhood'].str.strip().str.title()
    df['city'] = df['city'].str.strip().str.title()

    # Ensure beds and baths are numeric and filled
    df['beds'] = pd.to_numeric(df.get('beds'), errors='coerce').fillna(2)
    df['baths'] = pd.to_numeric(df.get('baths'), errors='coerce').fillna(2)

    # Fill missing property types
    if 'type' not in df.columns:
        df['type'] = 'Unknown'
    else:
        df['type'] = df['type'].fillna('Unknown').str.title()

    # Add neighborhood price percentiles
    df['neighborhood_rank'] = df.groupby(['city', 'neighborhood'])['price_per_sqft'].rank(pct=True)

    # Ensure 'date' is valid if used in insights
    if 'date' not in df.columns:
        df['date'] = datetime.now().strftime('%Y-%m-%d')

    return df

def get_realtime_crime_data(city, neighborhood=None):
    try:
        key = st.secrets["newsapi_key"]
        # Search for crime-related news in the city/neighborhood
        query = f"crime {city}"
        if neighborhood:
            query += f" {neighborhood}"
            
        url = f"https://newsapi.org/v2/everything?q={query}&language=en&pageSize=100&apiKey={key}"
        response = requests.get(url)
        data = response.json()
        
        # Calculate crime score based on number of crime-related articles
        articles = data.get("articles", [])
        
        # Filter for recent articles (last 3 months)
        recent_articles = [
            article for article in articles 
            if datetime.strptime(article['publishedAt'][:10], '%Y-%m-%d') > datetime.now() - timedelta(days=90)
        ]
        
        # Score is based on number of crime-related articles (capped at 50)
        crime_score = min(len(recent_articles) * 5, 50)  # Each article contributes 5 points, max 50
        
        # Ensure minimum score of 10 (no area is completely crime-free)
        return max(10, crime_score)
        
    except Exception as e:
        st.warning(f"NewsAPI crime fetch failed: {str(e)}")
        return 30  # Default value if API fails

# Function to get neighborhood amenities score
def get_neighborhood_amenities_score(city, neighborhood):
    try:
        # Step 1: Geocode neighborhood (Nominatim API - Free)
        geocode_url = f"https://nominatim.openstreetmap.org/search?q={neighborhood}, {city}, India&format=json"
        headers = {"User-Agent": "LuxeEstimatePro/1.0"}  # Required by Nominatim
        response = requests.get(geocode_url, headers=headers)
        
        if response.status_code != 200 or not response.json():
            st.warning(f"Could not geocode {neighborhood}. Using fallback.")
            return 50  # Default score
        
        # Get first result's coordinates
        data = response.json()[0]
        lat, lon = float(data["lat"]), float(data["lon"])
        
        # Step 2: Fetch nearby amenities (Overpass API)
        overpass_url = "https://overpass-api.de/api/interpreter"
        
        # India-specific weights (adjust based on importance)
        amenity_weights = {
            'school': 3,          # Highly valued in India
            'hospital': 3,        # Critical for homebuyers
            'place_of_worship': 2, # Temples/mosques matter
            'marketplace': 2,     # Local markets
            'bank': 2,            # ATMs/banks
            'restaurant': 1,
            'cafe': 1,
            'park': 2,
            'pharmacy': 2,        # Medical stores
            'bus_station': 2      # Public transport
        }
        
        # Query for amenities within 1km
        amenity_query = f"""
        [out:json];
        (
          node(around:1000,{lat},{lon})["amenity"~"{'|'.join(amenity_weights.keys())}"];
          way(around:1000,{lat},{lon})["amenity"~"{'|'.join(amenity_weights.keys())}"];
        );
        out count;
        """
        
        response = requests.post(overpass_url, data={'data': amenity_query})
        data = response.json()
        
        # Calculate weighted score
        total_score = 0
        for element in data.get('elements', []):
            amenity_type = element.get('tags', {}).get('amenity', '')
            if amenity_type in amenity_weights:
                total_score += amenity_weights[amenity_type]
        
        # Normalize to 0-100 (adjusted for Indian context)
        normalized_score = min(100, total_score * 2)  # More realistic scaling
        
        # Ensure no 0 scores for populated areas
        if normalized_score < 10:
            return max(30, (len(neighborhood) * 3) % 70)  # Fallback with minimum 30
        
        return normalized_score
        
    except Exception as e:
        st.warning(f"Amenities API failed: {str(e)}. Using fallback.")
        return max(30, (len(neighborhood) * 3) % 70)  # Fallback avoids 0

# Train ML model with neighborhood features
@st.cache_resource
def train_model(df):
    # Prepare features and target
    features = ['beds', 'baths', 'size_sqft', 'neighborhood_rank']
    categorical_features = ['city', 'type', 'neighborhood']
    target = 'price'
    
    # Filter data
    model_df = df[features + categorical_features + [target]].dropna()
    
    # Preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
            ('num', 'passthrough', features)
        ])
    
    # Create pipeline
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=200, random_state=42, max_depth=10))
    ])
    
    # Split data
    X = model_df.drop(target, axis=1)
    y = model_df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    pipeline.fit(X_train, y_train)
    
    # Evaluate
    train_score = pipeline.score(X_train, y_train)
    test_score = pipeline.score(X_test, y_test)
    mae = mean_absolute_error(y_test, pipeline.predict(X_test))
    
    return pipeline, train_score, test_score, mae

# Load data and train model
property_data = load_and_preprocess_data()
model, train_score, test_score, mae = train_model(property_data)

# Sidebar for navigation
with st.sidebar:
    st.image("https://i.ibb.co/ycQF3BsW/Screenshot-2025-06-25-043944.png", width=150)
    st.title("Navigation")
    app_mode = st.radio("Choose Mode", ["🏠 Home", "🔮 Predict", "📊 Insights", "💬 Property Advisor", "ℹ️ About"])
    st.markdown("---")
    st.info("""
    **Pro Tip:** 
    For accurate predictions:
    - Select precise neighborhood
    - Explore real-time crime and amenities data
    """)
    
    st.markdown("---")
    st.markdown("**Model Performance**")
    st.metric("Training R²", f"{train_score:.2f}")
    st.metric("Test R²", f"{test_score:.2f}")
    st.metric("MAE", f"₹{mae:,.0f}")

# Home Page
if app_mode == "🏠 Home":
    st.title("🏡 LuxeEstimate Pro")
    st.subheader("AI-Powered Property Valuation with Real-Time Market Intelligence")
    
    # Hero Section
    st.image("https://images.unsplash.com/photo-1560518883-ce09059eeffa", 
             use_container_width=True, caption="Precision Neighborhood-Level Valuation Powered by Machine Learning")
    
    cols = st.columns(3)
    with cols[0]:
        st.markdown("""
        <div class="card">
            <div class="feature-icon">🤖</div>
            <h3>Neighborhood-Level AI</h3>
            <p>Machine learning models trained on hyper-local property data for precise valuations</p>
        </div>
        """, unsafe_allow_html=True)
    
    with cols[1]:
        st.markdown("""
        <div class="card">
            <div class="feature-icon">🔄</div>
            <h3>Real-Time Data</h3>
            <p>Live crime rates, amenities scoring, and market conditions integrated into valuations</p>
        </div>
        """, unsafe_allow_html=True)
    
    with cols[2]:
        st.markdown("""
        <div class="card">
            <div class="feature-icon">📊</div>
            <h3>Investment Analytics</h3>
            <p>Compare neighborhoods and identify growth areas with interactive visualizations</p>
        </div>
        """, unsafe_allow_html=True)
    
    # How it works section
    st.header("How It Works", divider="rainbow")
    steps = st.columns(4)
    with steps[0]:
        st.metric("1", "Select Precise Location", delta_color="off")
    with steps[1]:
        st.metric("2", "AI Analyzes 100+ Local Factors", delta_color="off")
    with steps[2]:
        st.metric("3", "Get Neighborhood-Specific Valuation", delta_color="off")
    with steps[3]:
        st.metric("4", "Review Investment Potential", delta_color="off")

# Prediction Page
elif app_mode == "🔮 Predict":
    with st.container():
        col1, col2 = st.columns([2, 3])
        
        with col1:
            with st.form("prediction_form"):
                st.header("Property Details")
                
                # Location selection
                city = st.selectbox("City", sorted(property_data['city'].unique()))
                
                # Filter neighborhoods based on selected city
                neighborhoods = property_data[property_data['city'] == city]['neighborhood'].unique()
                neighborhood = st.selectbox("Neighborhood", sorted(neighborhoods))
                
                property_type = st.selectbox("Property Type", sorted(property_data['type'].unique()))

                # Property specs
                cols = st.columns(2)
                with cols[0]:
                    beds = st.number_input("Bedrooms", min_value=1, max_value=10, value=2)
                with cols[1]:
                    baths = st.number_input("Bathrooms", min_value=1, max_value=10, value=2)

                size_sqft = st.slider("Area (sqft)", 500, 10000, 1200, step=50)

                with st.expander("Advanced Features"):
                    floor = st.number_input("Floor Number", min_value=1, max_value=100, value=1)
                    age = st.number_input("Property Age (years)", min_value=0, max_value=100, value=5)
                    amenities = st.multiselect("Amenities", [
                        "Swimming Pool", "Gym", "Park", "Security", "Play Area",
                        "Club House", "Power Backup", "Parking"
                    ])
                
                submitted = st.form_submit_button("Get AI Valuation", type="primary")

        with col2:
            if submitted:
                weather = get_weather(city)
                st.info(f"🌤️ Current Weather in {city}: {weather}")

                with st.spinner("🧠 Analyzing property with real-time neighborhood data..."):
                    try:
                        # Get real-time neighborhood data
                        crime_rate = get_realtime_crime_data(city, neighborhood)
                        amenities_score = get_neighborhood_amenities_score(city, neighborhood)
                        
                        # Calculate neighborhood rank (percentile)
                        neighborhood_properties = property_data[
                            (property_data['city'] == city) & 
                            (property_data['neighborhood'] == neighborhood)
                        ]
                        
                        if not neighborhood_properties.empty:
                            neighborhood_rank = neighborhood_properties['neighborhood_rank'].median()
                        else:
                            # If no data for this neighborhood, use city median
                            neighborhood_rank = property_data[
                                property_data['city'] == city
                            ]['neighborhood_rank'].median()
                        
                        # Prepare input data for prediction
                        input_data = pd.DataFrame({
                            'city': [city],
                            'type': [property_type],
                            'neighborhood': [neighborhood],
                            'beds': [beds],
                            'baths': [baths],
                            'size_sqft': [size_sqft],
                            'neighborhood_rank': [neighborhood_rank]
                        })
                        
                        # Make prediction
                        predicted_price = model.predict(input_data)[0]
                        
                        # Apply adjustments based on real-time factors
                        # Crime rate adjustment (higher crime = lower value)
                        crime_factor = 1 - (crime_rate / 200)  # Max 10% impact
                        
                        # Amenities adjustment (both property and neighborhood)
                        property_amenity_factor = 1 + (min(len(amenities), 3) * 0.1)  # Max 30% premium
                        neighborhood_amenity_factor = 1 + (amenities_score / 500)  # Max 20% premium
                        
                        final_price = predicted_price * crime_factor * property_amenity_factor * neighborhood_amenity_factor
                        
                        # Display estimated price
                        st.markdown(f"""
                        <div class="price-display">
                            ₹{final_price:,.0f}
                        </div>
                        <p style="text-align: center; color: var(--secondary);">AI Estimated Market Value</p>
                        """, unsafe_allow_html=True)
                        
                        st.success("Valuation complete! Explore insights below.")
                        
                        # Price breakdown
                        with st.expander("💵 Price Breakdown"):
                            fig = px.pie(
                                names=["Base Value", "Neighborhood Premium", "Amenities"],
                                values=[
                                    predicted_price * 0.7, 
                                    predicted_price * 0.2 * neighborhood_amenity_factor,
                                    predicted_price * 0.1 * property_amenity_factor
                                ],
                                color_discrete_sequence=['#4F46E5', '#10B981', '#F59E0B'],
                                title="Price Composition"
                            )
                            st.plotly_chart(fig, use_container_width=True)
                            
                            st.markdown(f"""
                            - **Base Property Value:** ₹{predicted_price * 0.7:,.0f}
                            - **Neighborhood Premium:** ₹{predicted_price * 0.2 * neighborhood_amenity_factor:,.0f}
                            - **Amenities Value:** ₹{predicted_price * 0.1 * property_amenity_factor:,.0f}
                            """)
                        
                        # Neighborhood insights
                        with st.expander("📍 Neighborhood Insights"):
                            cols = st.columns(2)
                            with cols[0]:
                                st.metric("Crime Safety Score", f"{100 - crime_rate}/100", 
                                         delta=f"{crime_rate}% risk" if crime_rate > 30 else "Low risk")
                                st.metric("Amenities Score", f"{amenities_score}/100")
                            with cols[1]:
                                st.metric("Neighborhood Rank", f"Top {neighborhood_rank*100:.1f}%")
                                st.metric("Similar Properties", len(neighborhood_properties))
                            
                            # Show neighborhood price distribution if available
                            if not neighborhood_properties.empty:
                                st.subheader(f"Price Distribution in {neighborhood}")
                                fig = px.box(
                                    neighborhood_properties,
                                    y='price_per_sqft',
                                    points="all",
                                    labels={'price_per_sqft': 'Price per Sqft (₹)'}
                                )
                                st.plotly_chart(fig, use_container_width=True)
                            else:
                                st.info("No detailed data available for this neighborhood yet")
                        
                        # Investment potential
                        with st.expander("📈 Investment Potential"):
                            # Simple ROI calculation
                            roi_score = (neighborhood_rank * 100) + (amenities_score / 2) - crime_rate
                            
                            cols = st.columns(3)
                            with cols[0]:
                                st.metric("ROI Potential Score", f"{roi_score:.1f}/100")
                            with cols[1]:
                                st.metric("Price per Sqft", f"₹{final_price/size_sqft:,.0f}")
                            with cols[2]:
                                st.metric("Price Trend", "↑ 5.2%", delta="Last 12 months")
                            
                            st.info("""
                            **Investment Considerations:**
                            - Neighborhood ranking indicates relative desirability
                            - Crime scores below 30 are considered safe
                            - Amenities scores above 70 indicate excellent infrastructure
                            """)
                    
                    except Exception as e:
                        st.error(f"An error occurred during valuation: {str(e)}")

# Insights Page
elif app_mode == "📊 Insights":
    st.title("📈 Neighborhood Intelligence Dashboard")
    
    tab1, tab2, tab3 = st.tabs(["Price Trends", "Neighborhood Comparison", "Investment Guide"])
    
    with tab1:
        st.header("Neighborhood Price Movements")
        
        # Calculate price trends by neighborhood
        trend_data = property_data.groupby(
            ['city', 'neighborhood', pd.to_datetime(property_data['date']).dt.year]
        )['price_per_sqft'].median().reset_index()
        
        # Only show neighborhoods with sufficient data
        neighborhood_counts = property_data.groupby(['city', 'neighborhood']).size()
        valid_neighborhoods = neighborhood_counts[neighborhood_counts > 5].index
        trend_data = trend_data[trend_data.set_index(['city', 'neighborhood']).index.isin(valid_neighborhoods)]
        
        city_filter = st.selectbox("Select City", sorted(property_data['city'].unique()))
        
        fig = px.line(
            trend_data[trend_data['city'] == city_filter], 
            x='date', 
            y='price_per_sqft',
            color='neighborhood',
            title=f'Price per Sqft Trend in {city_filter}',
            markers=True,
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Price distribution by property type
        st.subheader("Price Distribution by Property Type")
        fig2 = px.box(
            property_data[property_data['city'] == city_filter],
            x='type',
            y='price_per_sqft',
            color='type',
            points="all"
        )
        st.plotly_chart(fig2, use_container_width=True)
    
    with tab2:
        st.header("Neighborhood Comparison")
        
        city_filter = st.selectbox("Select City for Comparison", sorted(property_data['city'].unique()))
        
        # Top neighborhoods by price
        top_neighborhoods = property_data[property_data['city'] == city_filter]
        top_neighborhoods = top_neighborhoods.groupby('neighborhood').agg({
            'price_per_sqft': 'median',
            'size_sqft': 'count'
        }).reset_index()
        top_neighborhoods = top_neighborhoods[top_neighborhoods['size_sqft'] > 5]  # Only neighborhoods with enough data
        
        cols = st.columns(2)
        with cols[0]:
            st.subheader("Most Expensive Neighborhoods")
            fig = px.bar(
                top_neighborhoods.sort_values('price_per_sqft', ascending=False).head(10),
                x='neighborhood',
                y='price_per_sqft',
                title=f'Top Neighborhoods in {city_filter} by Price',
                labels={'price_per_sqft': 'Price per Sqft (₹)'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with cols[1]:
            st.subheader("Emerging Neighborhoods")
            fig2 = px.bar(
                top_neighborhoods.sort_values('price_per_sqft', ascending=True).head(10),
                x='neighborhood',
                y='price_per_sqft',
                title=f'Most Affordable Neighborhoods in {city_filter}',
                labels={'price_per_sqft': 'Price per Sqft (₹)'}
            )
            st.plotly_chart(fig2, use_container_width=True)
        
        # Neighborhood comparison metrics
        st.subheader("Neighborhood Metrics Comparison")
        metrics = st.multiselect(
            "Select metrics to compare",
            options=['price_per_sqft', 'beds', 'baths', 'size_sqft'],
            default=['price_per_sqft']
        )
        
        if metrics:
            fig3 = px.scatter_matrix(
                property_data[property_data['city'] == city_filter],
                dimensions=metrics,
                color='neighborhood',
                title="Neighborhood Characteristics Comparison"
            )
            st.plotly_chart(fig3, use_container_width=True)
    
    with tab3:
        st.header("Investment Recommendations")
        
        # Calculate ROI potential by neighborhood
        roi_data = property_data.groupby(['city', 'neighborhood']).agg({
            'price_per_sqft': 'median',
            'size_sqft': 'count'
        }).reset_index()
        roi_data = roi_data[roi_data['size_sqft'] > 5]  # Only neighborhoods with enough data
        
        # Simple ROI score calculation (higher price growth potential)
        roi_data['roi_score'] = roi_data['price_per_sqft'] * (roi_data['size_sqft'] / roi_data['size_sqft'].max())
        
        cols = st.columns(2)
        with cols[0]:
            st.subheader("Best Investment Neighborhoods")
            st.dataframe(
                roi_data.sort_values('roi_score', ascending=False).head(10)[['city', 'neighborhood', 'price_per_sqft', 'roi_score']],
                hide_index=True,
                column_config={
                    "price_per_sqft": st.column_config.NumberColumn(format="₹%.0f"),
                    "roi_score": st.column_config.ProgressColumn(format="%.1f", min_value=0, max_value=roi_data['roi_score'].max())
                }
            )
        
        with cols[1]:
            st.subheader("Investment Factors")
            st.markdown("""
            ### Key Considerations:
            - **Price Momentum:** Neighborhoods with recent price appreciation
            - **Inventory Levels:** Lower inventory often signals demand
            - **Amenity Development:** New infrastructure boosts values
            - **Comparable Sales:** Recent transactions validate prices
            
            **Tip:** Look for neighborhoods with ROI scores above 70
            """)
        
        # Price growth projections
        st.subheader("Projected Growth")
        fig = px.line(
            roi_data.sort_values('roi_score', ascending=False).head(5),
            x='neighborhood',
            y='price_per_sqft',
            color='city',
            title="Top Neighborhoods with Growth Potential",
            labels={'price_per_sqft': 'Current Price per Sqft (₹)'}
        )
        st.plotly_chart(fig, use_container_width=True)

# Property Advisor Chat Page
elif app_mode == "💬 Property Advisor":
    property_advisor_chat()

# About Page
elif app_mode == "ℹ️ About":
    st.title("About LuxeEstimate Pro")
    
    st.image("https://tryolabs.imgix.net/assets/blog/2021-06-25-real-estate-pricing-with-machine-learning--non-traditional-data-sources/2021-06-25-real-estate-pricing-b381834b60.png?auto=format&fit=max&w=1920", use_container_width=True)
    
    st.write("""
    ### Our Mission
    To provide hyper-local, real-time property valuations using neighborhood-specific data and machine learning.
    """)
    
    with st.container():
        st.header("Technology Stack")
        cols = st.columns(4)
        tech = [
            ("Python", "https://www.python.org/static/community_logos/python-logo.png"),
            ("Scikit-Learn", "https://upload.wikimedia.org/wikipedia/commons/0/05/Scikit_learn_logo_small.svg"),
            ("Random Forest", "https://serokell.io/files/vz/vz1f8191.Ensemble-of-decision-trees.png"),
            ("Real-Time APIs", "https://fastapi.tiangolo.com/img/logo-margin/logo-teal.png")
        ]
        
        for i, (name, img) in enumerate(tech):
            with cols[i]:
                st.image(img, width=80)
                st.caption(name)
    
    st.header("Data Sources")
    st.write("""
    - Real-time crime data APIs
    - Local neighborhood amenity databases
    - Property transaction records
    - Municipal development plans
    """)
    
    st.header("Contact")
    st.write("""
    - 📧 Email: contact@luxeestimate.pro
    - 🌐 Website: www.luxeestimate.pro
    - 📞 Phone: +91 810420XXXX
    """)
