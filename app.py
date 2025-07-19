import streamlit as st
import requests
import os
import json
import groq
from dotenv import load_dotenv
from datetime import datetime, timedelta
import calendar
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# --- CONFIGURATION ---
load_dotenv()
NASA_API_KEY = os.getenv("NASA_API_KEY", "abjuRRnOFpcmDPzQlhCe6oUhSxyKy8oQS3IYUyEi")



# Initialize Groq client
@st.cache_resource
def init_groq_client():
    return groq.Groq(api_key=os.getenv("GROQ_API_KEY"))


# --- CORE FUNCTIONS ---

@st.cache_data(ttl=3600)
def determine_user_intent(user_query: str) -> dict | None:
    """Uses LLM to determine intent and extract parameters."""
    try:
        groq_client = init_groq_client()
        today_str = datetime.now().strftime('%Y-%m-%d')

        chat_completion = groq_client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": f"""You are an intelligent request router for agricultural and environmental analysis. Today's date is {today_str}.
Classify user queries into these intents:
- 'daily_weather_report': Daily weather for specific date
- 'farming_advice': Monthly farming advice
- 'flood_risk_assessment': Flood risk analysis for a location
- 'site_suitability_analysis': Agricultural site suitability assessment
- 'drought_monitoring': Drought condition monitoring
- 'soil_moisture_analysis': Soil moisture and agricultural conditions

Extract parameters:
- location: Always required
- date: For daily reports (YYYYMMDD format)
- month: For monthly analysis (YYYYMM format)
- period: For longer analysis (start_date and end_date in YYYYMMDD format)

Respond ONLY with JSON like: {{"intent": "intent_name", "params": {{"location": "...", "date": "...", "month": "...", "start_date": "...", "end_date": "..."}}}}"""
                },
                {"role": "user", "content": user_query}
            ],
            model="llama3-8b-8192", temperature=0.0
        )
        response_text = chat_completion.choices[0].message.content
        return json.loads(response_text)
    except Exception as e:
        st.error(f"Error determining intent: {e}")
        return None


@st.cache_data(ttl=3600)
def get_coordinates(location_name: str) -> dict | None:
    """Gets coordinates for a location."""
    try:
        groq_client = init_groq_client()
        chat_completion = groq_client.chat.completions.create(
            messages=[
                {"role": "system",
                 "content": "You are a geocoding expert. Given a location, respond ONLY with a JSON object like {\"lat\": <latitude>, \"lon\": <longitude>}."},
                {"role": "user", "content": location_name}
            ],
            model="llama3-8b-8192", temperature=0.0
        )
        return json.loads(chat_completion.choices[0].message.content)
    except Exception:
        return None


@st.cache_data(ttl=3600)
def fetch_weather_data(latitude: float, longitude: float, start_date: str, end_date: str) -> list | None:
    """Fetches hourly weather data for a date range from NASA POWER."""
    params = {
        "parameters": "T2M,RH2M,PRECTOTCORR", "community": "AG", "format": "JSON",
        "latitude": latitude, "longitude": longitude, "start": start_date, "end": end_date,
        "api_key": NASA_API_KEY,
    }
    try:
        response = requests.get("https://power.larc.nasa.gov/api/temporal/hourly/point", params=params)
        response.raise_for_status()
        data = response.json()
        if 'properties' not in data or 'parameter' not in data['properties']: return None
        parameters = data['properties']['parameter']
        if not parameters: return None
        processed_data, first_param_key = [], list(parameters.keys())[0]
        timestamps = list(parameters[first_param_key].keys())
        for ts in timestamps:
            entry = {'datetime': ts}
            for param, values in parameters.items():
                entry[param] = values.get(ts, 0) if values.get(ts, 0) != -999 else 0
            processed_data.append(entry)
        return processed_data
    except Exception as e:
        st.error(f"Error fetching weather data: {e}")
        return None


@st.cache_data(ttl=3600)
def fetch_extended_weather_data(latitude: float, longitude: float, start_date: str, end_date: str) -> list | None:
    """Fetches extended weather data including soil moisture, wind, and solar radiation."""
    params = {
        "parameters": "T2M,RH2M,PRECTOTCORR,WS2M,ALLSKY_SFC_SW_DWN,T2M_MAX,T2M_MIN,GWETROOT,GWETTOP",
        "community": "AG", "format": "JSON",
        "latitude": latitude, "longitude": longitude, "start": start_date, "end": end_date,
        "api_key": NASA_API_KEY,
    }
    try:
        response = requests.get("https://power.larc.nasa.gov/api/temporal/daily/point", params=params)
        response.raise_for_status()
        data = response.json()
        if 'properties' not in data or 'parameter' not in data['properties']: return None
        parameters = data['properties']['parameter']
        if not parameters: return None
        processed_data, first_param_key = [], list(parameters.keys())[0]
        timestamps = list(parameters[first_param_key].keys())
        for ts in timestamps:
            entry = {'date': ts}
            for param, values in parameters.items():
                entry[param] = values.get(ts, 0) if values.get(ts, 0) != -999 else 0
            processed_data.append(entry)
        return processed_data
    except Exception as e:
        st.error(f"Error fetching extended weather data: {e}")
        return None


def summarize_daily_weather(hourly_data: list) -> dict:
    """Summarizes hourly data into Morning, Afternoon, and Night periods."""
    periods = {
        "Morning (6am-12pm)": {"hours": range(6, 12), "T2M": [], "RH2M": [], "PRECTOTCORR": []},
        "Afternoon (12pm-6pm)": {"hours": range(12, 18), "T2M": [], "RH2M": [], "PRECTOTCORR": []},
        "Night (6pm-12am)": {"hours": range(18, 24), "T2M": [], "RH2M": [], "PRECTOTCORR": []},
    }
    for entry in hourly_data:
        hour = int(entry['datetime'][-2:])
        for period_name, data in periods.items():
            if hour in data['hours']:
                data["T2M"].append(entry.get("T2M", 0))
                data["RH2M"].append(entry.get("RH2M", 0))
                data["PRECTOTCORR"].append(entry.get("PRECTOTCORR", 0))
    summary = {}
    for period_name, data in periods.items():
        if not data["T2M"]:
            summary[period_name] = "No data available."
            continue
        avg_temp = sum(data["T2M"]) / len(data["T2M"])
        avg_humidity = sum(data["RH2M"]) / len(data["RH2M"])
        total_precip = sum(data["PRECTOTCORR"])
        summary[
            period_name] = f"Avg Temp: {avg_temp:.1f}°C, Avg Humidity: {avg_humidity:.1f}%, Total Rain: {total_precip:.2f} mm"
    return summary


@st.cache_data(ttl=3600)
def get_farming_advice_from_llm(user_query: str, monthly_data: list) -> str:
    """Uses LLM to provide farming advice based on weather data."""
    data_str = json.dumps(monthly_data)
    try:
        groq_client = init_groq_client()
        chat_completion = groq_client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are an agricultural expert. Analyze the provided weather data to answer farming questions. Provide actionable recommendations based on temperature, humidity, precipitation, and soil moisture conditions."
                },
                {"role": "user",
                 "content": f"Weather data:\n{data_str}\n\nQuestion: {user_query}"}
            ],
            model="meta-llama/llama-4-scout-17b-16e-instruct",
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        return f"Error generating advice: {e}"


# --- NEW ADVANCED ANALYSIS FUNCTIONS ---

def assess_flood_risk(weather_data: list, location: str) -> dict:
    """Assesses flood risk based on precipitation patterns and soil moisture."""
    if not weather_data:
        return {"risk_level": "Unknown", "analysis": "No data available"}

    # Calculate precipitation statistics
    precip_values = [entry.get("PRECTOTCORR", 0) for entry in weather_data]
    total_precip = sum(precip_values)
    avg_precip = total_precip / len(precip_values) if precip_values else 0
    max_precip = max(precip_values) if precip_values else 0

    # Calculate consecutive high precipitation days (adjusted threshold)
    consecutive_high_precip = 0
    max_consecutive = 0
    threshold = 15  # mm per day (increased from 10mm for more realistic assessment)

    for precip in precip_values:
        if precip > threshold:
            consecutive_high_precip += 1
            max_consecutive = max(max_consecutive, consecutive_high_precip)
        else:
            consecutive_high_precip = 0

    # Calculate days with moderate precipitation (5-15mm)
    moderate_precip_days = sum(1 for p in precip_values if 5 <= p <= 15)

    # Calculate days with heavy precipitation (>25mm)
    heavy_precip_days = sum(1 for p in precip_values if p > 25)

    # Get soil moisture data if available
    soil_moisture = [entry.get("GWETTOP", 0) for entry in weather_data]
    avg_soil_moisture = sum(soil_moisture) / len(soil_moisture) if soil_moisture else 0

    # More realistic risk assessment logic
    risk_score = 0

    # 1. Total precipitation factor (adjusted thresholds)
    days_analyzed = len(precip_values)
    if days_analyzed >= 30:  # Monthly analysis
        if total_precip > 400:  # Very high total precipitation for a month
            risk_score += 4
        elif total_precip > 250:  # High precipitation
            risk_score += 3
        elif total_precip > 150:  # Moderate precipitation
            risk_score += 2
        elif total_precip > 80:  # Light precipitation
            risk_score += 1
    else:  # Shorter period analysis - scale proportionally
        scaled_threshold = (400 * days_analyzed) / 30
        if total_precip > scaled_threshold:
            risk_score += 4
        elif total_precip > scaled_threshold * 0.625:  # 250/400
            risk_score += 3
        elif total_precip > scaled_threshold * 0.375:  # 150/400
            risk_score += 2
        elif total_precip > scaled_threshold * 0.2:  # 80/400
            risk_score += 1

    # 2. Consecutive high precipitation days factor (more strict)
    if max_consecutive >= 7:  # 7+ consecutive days of heavy rain
        risk_score += 4
    elif max_consecutive >= 5:  # 5-6 consecutive days
        risk_score += 3
    elif max_consecutive >= 3:  # 3-4 consecutive days
        risk_score += 2
    elif max_consecutive >= 2:  # 2 consecutive days
        risk_score += 1

    # 3. Heavy precipitation days factor (new)
    heavy_precip_ratio = heavy_precip_days / days_analyzed
    if heavy_precip_ratio > 0.3:  # More than 30% of days had heavy rain
        risk_score += 3
    elif heavy_precip_ratio > 0.2:  # 20-30% of days
        risk_score += 2
    elif heavy_precip_ratio > 0.1:  # 10-20% of days
        risk_score += 1

    # 4. Soil moisture factor (adjusted thresholds)
    if avg_soil_moisture > 0.9:  # Extremely saturated soil
        risk_score += 3
    elif avg_soil_moisture > 0.8:  # Very saturated soil
        risk_score += 2
    elif avg_soil_moisture > 0.7:  # Moderately saturated soil
        risk_score += 1

    # 5. Maximum daily precipitation factor (new)
    if max_precip > 50:  # Extreme daily precipitation
        risk_score += 3
    elif max_precip > 30:  # Very high daily precipitation
        risk_score += 2
    elif max_precip > 20:  # High daily precipitation
        risk_score += 1

    # 6. Precipitation intensity factor (average daily precipitation)
    if avg_precip > 15:  # High average daily precipitation
        risk_score += 2
    elif avg_precip > 10:  # Moderate average daily precipitation
        risk_score += 1

    # Determine risk level with adjusted thresholds
    if risk_score >= 12:  # Very high threshold
        risk_level = "High"
        color = "red"
    elif risk_score >= 8:  # High threshold
        risk_level = "Moderate"
        color = "orange"
    elif risk_score >= 4:  # Moderate threshold
        risk_level = "Low"
        color = "yellow"
    else:
        risk_level = "Very Low"
        color = "green"

    # Generate detailed analysis
    analysis_parts = []
    analysis_parts.append(f"Based on {days_analyzed} days of data")
    analysis_parts.append(f"Total precipitation: {total_precip:.1f}mm")
    analysis_parts.append(f"Average daily precipitation: {avg_precip:.1f}mm")
    analysis_parts.append(f"Maximum daily precipitation: {max_precip:.1f}mm")
    analysis_parts.append(f"Heavy precipitation days: {heavy_precip_days}")
    analysis_parts.append(f"Max consecutive high-precip days: {max_consecutive}")
    analysis_parts.append(f"Average soil moisture: {avg_soil_moisture:.2f}")

    detailed_analysis = f"Flood risk assessment: {risk_level} (Score: {risk_score}). " + ". ".join(analysis_parts)

    return {
        "risk_level": risk_level,
        "risk_score": risk_score,
        "color": color,
        "total_precipitation": total_precip,
        "average_daily_precip": avg_precip,
        "max_daily_precip": max_precip,
        "heavy_precip_days": heavy_precip_days,
        "moderate_precip_days": moderate_precip_days,
        "consecutive_high_precip_days": max_consecutive,
        "avg_soil_moisture": avg_soil_moisture,
        "days_analyzed": days_analyzed,
        "analysis": detailed_analysis
    }


def create_improved_flood_visualization(weather_data: list) -> go.Figure:
    """Creates an improved flood risk visualization with better insights."""
    if not weather_data:
        return go.Figure()

    df = pd.DataFrame(weather_data)

    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Daily Precipitation with Risk Thresholds',
            'Soil Moisture Levels',
            'Cumulative Precipitation',
            'Precipitation Distribution'
        ),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"type": "histogram"}]]
    )

    # 1. Daily Precipitation with thresholds
    fig.add_trace(go.Scatter(
        x=df['date'],
        y=df['PRECTOTCORR'],
        mode='lines+markers',
        name='Daily Precipitation',
        line=dict(color='blue'),
        marker=dict(size=4)
    ), row=1, col=1)

    # Add risk threshold lines
    fig.add_hline(y=15, line_dash="dash", line_color="orange",
                  annotation_text="High Risk Threshold (15mm)", row=1, col=1)
    fig.add_hline(y=25, line_dash="dash", line_color="red",
                  annotation_text="Very High Risk (25mm)", row=1, col=1)

    # 2. Soil Moisture with optimal ranges
    fig.add_trace(go.Scatter(
        x=df['date'],
        y=df['GWETTOP'],
        mode='lines+markers',
        name='Soil Moisture',
        line=dict(color='brown'),
        fill='tonexty'
    ), row=1, col=2)

    fig.add_hline(y=0.8, line_dash="dash", line_color="orange",
                  annotation_text="Saturation Risk (0.8)", row=1, col=2)
    fig.add_hline(y=0.9, line_dash="dash", line_color="red",
                  annotation_text="High Saturation (0.9)", row=1, col=2)

    # 3. Cumulative Precipitation
    df['cumulative_precip'] = df['PRECTOTCORR'].cumsum()
    fig.add_trace(go.Scatter(
        x=df['date'],
        y=df['cumulative_precip'],
        mode='lines',
        name='Cumulative Precipitation',
        line=dict(color='darkblue', width=3)
    ), row=2, col=1)

    # 4. Precipitation Distribution (Histogram)
    fig.add_trace(go.Histogram(
        x=df['PRECTOTCORR'],
        nbinsx=20,
        name='Precipitation Distribution',
        marker_color='lightblue'
    ), row=2, col=2)

    # Update layout
    fig.update_layout(
        height=700,
        showlegend=True,
        title_text="Comprehensive Flood Risk Analysis"
    )

    # Update axes labels
    fig.update_xaxes(title_text="Date", row=1, col=1)
    fig.update_yaxes(title_text="Precipitation (mm)", row=1, col=1)
    fig.update_xaxes(title_text="Date", row=1, col=2)
    fig.update_yaxes(title_text="Soil Moisture", row=1, col=2)
    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_yaxes(title_text="Cumulative Precipitation (mm)", row=2, col=1)
    fig.update_xaxes(title_text="Daily Precipitation (mm)", row=2, col=2)
    fig.update_yaxes(title_text="Frequency", row=2, col=2)

    return fig

def analyze_site_suitability(weather_data: list, location: str) -> dict:
    """Analyzes agricultural site suitability based on multiple factors."""
    if not weather_data:
        return {"suitability": "Unknown", "analysis": "No data available"}

    # Extract relevant parameters
    temperatures = [entry.get("T2M", 0) for entry in weather_data]
    precipitation = [entry.get("PRECTOTCORR", 0) for entry in weather_data]
    humidity = [entry.get("RH2M", 0) for entry in weather_data]
    solar_radiation = [entry.get("ALLSKY_SFC_SW_DWN", 0) for entry in weather_data]
    soil_moisture = [entry.get("GWETTOP", 0) for entry in weather_data]

    # Calculate statistics
    avg_temp = sum(temperatures) / len(temperatures) if temperatures else 0
    total_precip = sum(precipitation)
    avg_humidity = sum(humidity) / len(humidity) if humidity else 0
    avg_solar = sum(solar_radiation) / len(solar_radiation) if solar_radiation else 0
    avg_soil_moisture = sum(soil_moisture) / len(soil_moisture) if soil_moisture else 0

    # Suitability scoring
    suitability_score = 0
    factors = []

    # Temperature suitability (optimal range 20-30°C)
    if 20 <= avg_temp <= 30:
        suitability_score += 3
        factors.append("Optimal temperature range")
    elif 15 <= avg_temp <= 35:
        suitability_score += 2
        factors.append("Good temperature range")
    else:
        suitability_score += 1
        factors.append("Suboptimal temperature")

    # Precipitation suitability (optimal 500-1500mm annually)
    annual_precip = total_precip * (365 / len(precipitation))
    if 500 <= annual_precip <= 1500:
        suitability_score += 3
        factors.append("Optimal precipitation levels")
    elif 300 <= annual_precip <= 2000:
        suitability_score += 2
        factors.append("Adequate precipitation")
    else:
        suitability_score += 1
        factors.append("Challenging precipitation levels")

    # Solar radiation suitability
    if avg_solar >= 15:
        suitability_score += 3
        factors.append("Excellent solar radiation")
    elif avg_solar >= 10:
        suitability_score += 2
        factors.append("Good solar radiation")
    else:
        suitability_score += 1
        factors.append("Limited solar radiation")

    # Soil moisture suitability
    if 0.3 <= avg_soil_moisture <= 0.7:
        suitability_score += 3
        factors.append("Optimal soil moisture")
    elif 0.2 <= avg_soil_moisture <= 0.8:
        suitability_score += 2
        factors.append("Good soil moisture")
    else:
        suitability_score += 1
        factors.append("Challenging soil moisture")

    # Determine suitability level
    max_score = 12
    if suitability_score >= 10:
        suitability = "Excellent"
        color = "green"
    elif suitability_score >= 8:
        suitability = "Good"
        color = "lightgreen"
    elif suitability_score >= 6:
        suitability = "Moderate"
        color = "yellow"
    else:
        suitability = "Poor"
        color = "red"

    return {
        "suitability": suitability,
        "score": suitability_score,
        "max_score": max_score,
        "color": color,
        "factors": factors,
        "avg_temperature": avg_temp,
        "annual_precipitation": annual_precip,
        "avg_solar_radiation": avg_solar,
        "avg_soil_moisture": avg_soil_moisture,
        "analysis": f"Site suitability is {suitability} ({suitability_score}/{max_score}). Key factors: {', '.join(factors[:3])}"
    }


def analyze_drought_conditions(weather_data: list, location: str) -> dict:
    """Analyzes drought conditions based on precipitation and soil moisture."""
    if not weather_data:
        return {"drought_level": "Unknown", "analysis": "No data available"}

    # Extract relevant data
    precipitation = [entry.get("PRECTOTCORR", 0) for entry in weather_data]
    soil_moisture = [entry.get("GWETTOP", 0) for entry in weather_data]
    temperatures = [entry.get("T2M", 0) for entry in weather_data]

    # Calculate drought indicators
    total_precip = sum(precipitation)
    avg_soil_moisture = sum(soil_moisture) / len(soil_moisture) if soil_moisture else 0
    avg_temp = sum(temperatures) / len(temperatures) if temperatures else 0

    # Count dry days (precipitation < 1mm)
    dry_days = sum(1 for p in precipitation if p < 1)
    dry_day_percentage = (dry_days / len(precipitation)) * 100

    # Drought severity calculation
    drought_score = 0

    # Precipitation factor
    if total_precip < 25:  # Very low precipitation
        drought_score += 4
    elif total_precip < 50:
        drought_score += 3
    elif total_precip < 100:
        drought_score += 2
    elif total_precip < 150:
        drought_score += 1

    # Soil moisture factor
    if avg_soil_moisture < 0.2:
        drought_score += 4
    elif avg_soil_moisture < 0.3:
        drought_score += 3
    elif avg_soil_moisture < 0.4:
        drought_score += 2
    elif avg_soil_moisture < 0.5:
        drought_score += 1

    # Temperature factor (higher temps increase drought risk)
    if avg_temp > 35:
        drought_score += 2
    elif avg_temp > 30:
        drought_score += 1

    # Dry days factor
    if dry_day_percentage > 80:
        drought_score += 2
    elif dry_day_percentage > 60:
        drought_score += 1

    # Determine drought level
    if drought_score >= 8:
        drought_level = "Severe"
        color = "red"
    elif drought_score >= 6:
        drought_level = "Moderate"
        color = "orange"
    elif drought_score >= 4:
        drought_level = "Mild"
        color = "yellow"
    else:
        drought_level = "None"
        color = "green"

    return {
        "drought_level": drought_level,
        "drought_score": drought_score,
        "color": color,
        "total_precipitation": total_precip,
        "avg_soil_moisture": avg_soil_moisture,
        "dry_days": dry_days,
        "dry_day_percentage": dry_day_percentage,
        "avg_temperature": avg_temp,
        "analysis": f"Drought level: {drought_level}. {dry_days} dry days ({dry_day_percentage:.1f}%) out of {len(precipitation)} days analyzed."
    }


def create_weather_visualization(weather_data: list, analysis_type: str) -> go.Figure:
    """Creates interactive visualizations for weather data."""
    if not weather_data:
        return go.Figure()

    df = pd.DataFrame(weather_data)

    if analysis_type == "flood_risk":
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Daily Precipitation', 'Soil Moisture', 'Temperature', 'Cumulative Precipitation'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )

        # Precipitation
        fig.add_trace(go.Scatter(
            x=df['date'], y=df['PRECTOTCORR'],
            mode='lines+markers', name='Precipitation (mm)',
            line=dict(color='blue')
        ), row=1, col=1)

        # Soil Moisture
        fig.add_trace(go.Scatter(
            x=df['date'], y=df['GWETTOP'],
            mode='lines+markers', name='Soil Moisture',
            line=dict(color='brown')
        ), row=1, col=2)

        # Temperature
        fig.add_trace(go.Scatter(
            x=df['date'], y=df['T2M'],
            mode='lines+markers', name='Temperature (°C)',
            line=dict(color='red')
        ), row=2, col=1)

        # Cumulative Precipitation
        df['cumulative_precip'] = df['PRECTOTCORR'].cumsum()
        fig.add_trace(go.Scatter(
            x=df['date'], y=df['cumulative_precip'],
            mode='lines', name='Cumulative Precipitation',
            line=dict(color='darkblue')
        ), row=2, col=2)

    elif analysis_type == "site_suitability":
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Temperature Range', 'Precipitation', 'Solar Radiation', 'Soil Moisture'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )

        # Temperature with min/max
        fig.add_trace(go.Scatter(
            x=df['date'], y=df['T2M_MAX'],
            mode='lines', name='Max Temp', line=dict(color='red', dash='dash')
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=df['date'], y=df['T2M_MIN'],
            mode='lines', name='Min Temp', line=dict(color='blue', dash='dash')
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=df['date'], y=df['T2M'],
            mode='lines', name='Avg Temp', line=dict(color='orange')
        ), row=1, col=1)

        # Precipitation
        fig.add_trace(go.Bar(
            x=df['date'], y=df['PRECTOTCORR'],
            name='Precipitation', marker_color='blue'
        ), row=1, col=2)

        # Solar Radiation
        fig.add_trace(go.Scatter(
            x=df['date'], y=df['ALLSKY_SFC_SW_DWN'],
            mode='lines+markers', name='Solar Radiation',
            line=dict(color='yellow')
        ), row=2, col=1)

        # Soil Moisture
        fig.add_trace(go.Scatter(
            x=df['date'], y=df['GWETTOP'],
            mode='lines+markers', name='Soil Moisture',
            line=dict(color='brown')
        ), row=2, col=2)

    fig.update_layout(height=600, showlegend=True)
    return fig


# --- STREAMLIT APP ---
def main():
    st.set_page_config(
        page_title="Advanced Agricultural Intelligence System",
        page_icon="🌾",
        layout="wide"
    )

    st.title("🌾 Advanced Agricultural Intelligence System")
    st.markdown(
        "Get weather reports, farming advice, flood risk assessment, site suitability analysis, and drought monitoring")

    # Sidebar for app info
    with st.sidebar:
        st.header("🚀 Available Features")
        st.markdown("""
        **Weather & Climate:**
        - Daily weather reports
        - Monthly farming advice

        **Risk Assessment:**
        - Flood risk analysis
        - Drought condition monitoring

        **Agricultural Planning:**
        - Site suitability analysis
        - Crop planning recommendations
        """)

        st.header("📝 Example Queries")
        st.markdown("""
        - "Weather in Delhi yesterday"
        - "Flood risk assessment for Mumbai this month"
        - "Is this site suitable for farming in California?"
        - "Drought conditions in Texas this season"
        - "Should I plant rice in Punjab this August?"
        """)

        st.header("🔧 Settings")
        if st.button("Clear Cache"):
            st.cache_data.clear()
            st.success("Cache cleared!")

    # Main interface
    col1, col2 = st.columns([3, 1])

    with col1:
        user_query = st.text_input(
            "Ask about weather, farming, floods, drought, or site suitability:",
            placeholder="e.g., 'flood risk assessment for Mumbai this month'"
        )

    with col2:
        search_button = st.button("🔍 Analyze", type="primary")

    if search_button and user_query:
        with st.spinner("🧠 Analyzing your question..."):
            intent_data = determine_user_intent(user_query)

            if not intent_data or "intent" not in intent_data:
                st.error("I'm sorry, I couldn't understand that. Please rephrase your question.")
                return

            intent = intent_data["intent"]
            params = intent_data["params"]
            location = params.get("location")

            if not location:
                st.error("Please specify a location in your query.")
                return

            with st.spinner(f"📍 Finding coordinates for {location}..."):
                coords = get_coordinates(location)

                if not coords:
                    st.error(f"Sorry, I couldn't find a location named '{location}'.")
                    return

            # Handle different analysis types
            if intent == 'daily_weather_report':
                date = params.get('date')

                with st.spinner("🛰️ Fetching weather data..."):
                    weather_data = fetch_weather_data(coords['lat'], coords['lon'], date, date)

                    if weather_data:
                        summary = summarize_daily_weather(weather_data)
                        date_obj = datetime.strptime(date, '%Y%m%d')
                        formatted_date = date_obj.strftime('%B %d, %Y')

                        st.success(f"Weather data found for {location.title()}!")

                        # Display weather summary
                        st.subheader(f"🌤️ Weather Summary for {location.title()} on {formatted_date}")

                        col1, col2, col3 = st.columns(3)

                        periods = ["Morning (6am-12pm)", "Afternoon (12pm-6pm)", "Night (6pm-12am)"]
                        icons = ["🌅", "☀️", "🌙"]

                        for i, (period, icon) in enumerate(zip(periods, icons)):
                            with [col1, col2, col3][i]:
                                st.info(f"**{icon} {period}**\n\n{summary[period]}")
                    else:
                        st.error(f"Sorry, no data found for {location.title()} on {date}.")

            elif intent == 'farming_advice':
                month_str = params.get('month')
                year, month = int(month_str[:4]), int(month_str[4:])
                start_date = f"{year}{month:02d}01"
                num_days = calendar.monthrange(year, month)[1]
                end_date = f"{year}{month:02d}{num_days}"

                with st.spinner("🛰️ Fetching monthly weather data..."):
                    weather_data = fetch_extended_weather_data(coords['lat'], coords['lon'], start_date, end_date)

                    if weather_data:
                        with st.spinner("🌾 Analyzing data for farming advice..."):
                            advice = get_farming_advice_from_llm(user_query, weather_data)

                            st.success(f"Farming advice for {location.title()}!")

                            # Display farming advice
                            st.subheader(f"🌾 Farming Advice for {datetime(year, month, 1).strftime('%B %Y')}")
                            st.markdown(advice)

                            # Display detailed statistics
                            with st.expander("📊 Detailed Weather Statistics"):
                                temps = [entry.get("T2M", 0) for entry in weather_data]
                                humidity = [entry.get("RH2M", 0) for entry in weather_data]
                                precip = [entry.get("PRECTOTCORR", 0) for entry in weather_data]
                                soil_moisture = [entry.get("GWETTOP", 0) for entry in weather_data]

                                col1, col2, col3, col4 = st.columns(4)

                                with col1:
                                    st.metric("Avg Temperature", f"{sum(temps) / len(temps):.1f}°C")
                                with col2:
                                    st.metric("Total Rainfall", f"{sum(precip):.2f} mm")
                                with col3:
                                    st.metric("Avg Humidity", f"{sum(humidity) / len(humidity):.1f}%")
                                with col4:
                                    st.metric("Avg Soil Moisture", f"{sum(soil_moisture) / len(soil_moisture):.2f}")
                    else:
                        st.error(
                            f"Sorry, I couldn't get enough data for {location.title()} in {datetime(year, month, 1).strftime('%B %Y')} to give advice.")

            elif intent == 'flood_risk_assessment':
                # Get date range (default to last 30 days)
                end_date = datetime.now()
                start_date = end_date - timedelta(days=30)
                start_date_str = start_date.strftime('%Y%m%d')
                end_date_str = end_date.strftime('%Y%m%d')

                # Override with user-specified dates if available
                if params.get('start_date') and params.get('end_date'):
                    start_date_str = params['start_date']
                    end_date_str = params['end_date']
                elif params.get('month'):
                    month_str = params['month']
                    year, month = int(month_str[:4]), int(month_str[4:])
                    start_date_str = f"{year}{month:02d}01"
                    num_days = calendar.monthrange(year, month)[1]
                    end_date_str = f"{year}{month:02d}{num_days}"

                with st.spinner("🌊 Analyzing flood risk..."):
                    weather_data = fetch_extended_weather_data(coords['lat'], coords['lon'], start_date_str,
                                                               end_date_str)

                    if weather_data:
                        flood_analysis = assess_flood_risk(weather_data, location)

                        st.success(f"Flood risk assessment complete for {location.title()}!")

                        # Display flood risk results
                        st.subheader("🌊 Flood Risk Assessment")

                        # Risk level indicator
                        col1, col2, col3 = st.columns([1, 2, 1])
                        with col2:
                            if flood_analysis['color'] == 'red':
                                st.error(f"⚠️ **FLOOD RISK: {flood_analysis['risk_level'].upper()}**")
                            elif flood_analysis['color'] == 'orange':
                                st.warning(f"⚠️ **FLOOD RISK: {flood_analysis['risk_level'].upper()}**")
                            elif flood_analysis['color'] == 'yellow':
                                st.warning(f"⚠️ **FLOOD RISK: {flood_analysis['risk_level'].upper()}**")
                            else:
                                st.success(f"✅ **FLOOD RISK: {flood_analysis['risk_level'].upper()}**")

                        # Detailed metrics
                        col1, col2, col3, col4 = st.columns(4)

                        with col1:
                            st.metric("Total Precipitation", f"{flood_analysis['total_precipitation']:.1f} mm")
                        with col2:
                            st.metric("Max Daily Precip", f"{flood_analysis['max_daily_precip']:.1f} mm")
                        with col3:
                            st.metric("Consecutive High-Precip Days",
                                      f"{flood_analysis['consecutive_high_precip_days']}")
                        with col4:
                            st.metric("Avg Soil Moisture", f"{flood_analysis['avg_soil_moisture']:.2f}")

                        # Analysis details
                        st.markdown("**Analysis:**")
                        st.info(flood_analysis['analysis'])

                        # Visualization
                        with st.expander("📈 Flood Risk Visualization"):
                            fig = create_weather_visualization(weather_data, "flood_risk")
                            st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.error(f"Sorry, couldn't fetch flood risk data for {location.title()}.")

            elif intent == 'site_suitability_analysis':
                # Get date range (default to last 90 days for better analysis)
                end_date = datetime.now()
                start_date = end_date - timedelta(days=90)
                start_date_str = start_date.strftime('%Y%m%d')
                end_date_str = end_date.strftime('%Y%m%d')

                # Override with user-specified dates if available
                if params.get('start_date') and params.get('end_date'):
                    start_date_str = params['start_date']
                    end_date_str = params['end_date']
                elif params.get('month'):
                    month_str = params['month']
                    year, month = int(month_str[:4]), int(month_str[4:])
                    start_date_str = f"{year}{month:02d}01"
                    num_days = calendar.monthrange(year, month)[1]
                    end_date_str = f"{year}{month:02d}{num_days}"

                with st.spinner("🏞️ Analyzing site suitability..."):
                    weather_data = fetch_extended_weather_data(coords['lat'], coords['lon'], start_date_str,
                                                               end_date_str)

                    if weather_data:
                        suitability_analysis = analyze_site_suitability(weather_data, location)

                        st.success(f"Site suitability analysis complete for {location.title()}!")

                        # Display suitability results
                        st.subheader("🏞️ Agricultural Site Suitability Analysis")

                        # Suitability level indicator
                        col1, col2, col3 = st.columns([1, 2, 1])
                        with col2:
                            if suitability_analysis['color'] == 'green':
                                st.success(f"✅ **SUITABILITY: {suitability_analysis['suitability'].upper()}**")
                            elif suitability_analysis['color'] == 'lightgreen':
                                st.success(f"✅ **SUITABILITY: {suitability_analysis['suitability'].upper()}**")
                            elif suitability_analysis['color'] == 'yellow':
                                st.warning(f"⚠️ **SUITABILITY: {suitability_analysis['suitability'].upper()}**")
                            else:
                                st.error(f"❌ **SUITABILITY: {suitability_analysis['suitability'].upper()}**")

                        # Score display
                        st.progress(suitability_analysis['score'] / suitability_analysis['max_score'])
                        st.caption(
                            f"Suitability Score: {suitability_analysis['score']}/{suitability_analysis['max_score']}")

                        # Detailed metrics
                        col1, col2, col3, col4 = st.columns(4)

                        with col1:
                            st.metric("Avg Temperature", f"{suitability_analysis['avg_temperature']:.1f}°C")
                        with col2:
                            st.metric("Annual Precipitation", f"{suitability_analysis['annual_precipitation']:.0f} mm")
                        with col3:
                            st.metric("Avg Solar Radiation",
                                      f"{suitability_analysis['avg_solar_radiation']:.1f} MJ/m²/day")
                        with col4:
                            st.metric("Avg Soil Moisture", f"{suitability_analysis['avg_soil_moisture']:.2f}")

                        # Key factors
                        st.markdown("**Key Factors:**")
                        for factor in suitability_analysis['factors']:
                            st.write(f"• {factor}")

                        # Analysis details
                        st.markdown("**Analysis:**")
                        st.info(suitability_analysis['analysis'])

                        # Visualization
                        with st.expander("📈 Site Suitability Visualization"):
                            fig = create_weather_visualization(weather_data, "site_suitability")
                            st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.error(f"Sorry, couldn't fetch site suitability data for {location.title()}.")

            elif intent == 'drought_monitoring':
                # Get date range (default to last 60 days)
                end_date = datetime.now()
                start_date = end_date - timedelta(days=60)
                start_date_str = start_date.strftime('%Y%m%d')
                end_date_str = end_date.strftime('%Y%m%d')

                # Override with user-specified dates if available
                if params.get('start_date') and params.get('end_date'):
                    start_date_str = params['start_date']
                    end_date_str = params['end_date']
                elif params.get('month'):
                    month_str = params['month']
                    year, month = int(month_str[:4]), int(month_str[4:])
                    start_date_str = f"{year}{month:02d}01"
                    num_days = calendar.monthrange(year, month)[1]
                    end_date_str = f"{year}{month:02d}{num_days}"

                with st.spinner("🌵 Analyzing drought conditions..."):
                    weather_data = fetch_extended_weather_data(coords['lat'], coords['lon'], start_date_str,
                                                               end_date_str)

                    if weather_data:
                        drought_analysis = analyze_drought_conditions(weather_data, location)

                        st.success(f"Drought monitoring complete for {location.title()}!")

                        # Display drought results
                        st.subheader("🌵 Drought Condition Monitoring")

                        # Drought level indicator
                        col1, col2, col3 = st.columns([1, 2, 1])
                        with col2:
                            if drought_analysis['color'] == 'red':
                                st.error(f"🚨 **DROUGHT LEVEL: {drought_analysis['drought_level'].upper()}**")
                            elif drought_analysis['color'] == 'orange':
                                st.warning(f"⚠️ **DROUGHT LEVEL: {drought_analysis['drought_level'].upper()}**")
                            elif drought_analysis['color'] == 'yellow':
                                st.warning(f"⚠️ **DROUGHT LEVEL: {drought_analysis['drought_level'].upper()}**")
                            else:
                                st.success(f"✅ **DROUGHT LEVEL: {drought_analysis['drought_level'].upper()}**")

                        # Detailed metrics
                        col1, col2, col3, col4 = st.columns(4)

                        with col1:
                            st.metric("Total Precipitation", f"{drought_analysis['total_precipitation']:.1f} mm")
                        with col2:
                            st.metric("Avg Soil Moisture", f"{drought_analysis['avg_soil_moisture']:.2f}")
                        with col3:
                            st.metric("Dry Days", f"{drought_analysis['dry_days']}")
                        with col4:
                            st.metric("Avg Temperature", f"{drought_analysis['avg_temperature']:.1f}°C")

                        # Additional metrics
                        st.markdown("**Drought Indicators:**")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Dry Day Percentage", f"{drought_analysis['dry_day_percentage']:.1f}%")
                        with col2:
                            st.metric("Drought Score", f"{drought_analysis['drought_score']}/12")

                        # Analysis details
                        st.markdown("**Analysis:**")
                        st.info(drought_analysis['analysis'])

                        # Visualization
                        with st.expander("📈 Drought Monitoring Visualization"):
                            fig = create_weather_visualization(weather_data, "flood_risk")  # Same viz works for drought
                            st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.error(f"Sorry, couldn't fetch drought monitoring data for {location.title()}.")

            elif intent == 'soil_moisture_analysis':
                # Get date range (default to last 30 days)
                end_date = datetime.now()
                start_date = end_date - timedelta(days=30)
                start_date_str = start_date.strftime('%Y%m%d')
                end_date_str = end_date.strftime('%Y%m%d')

                # Override with user-specified dates if available
                if params.get('start_date') and params.get('end_date'):
                    start_date_str = params['start_date']
                    end_date_str = params['end_date']
                elif params.get('month'):
                    month_str = params['month']
                    year, month = int(month_str[:4]), int(month_str[4:])
                    start_date_str = f"{year}{month:02d}01"
                    num_days = calendar.monthrange(year, month)[1]
                    end_date_str = f"{year}{month:02d}{num_days}"

                with st.spinner("💧 Analyzing soil moisture conditions..."):
                    weather_data = fetch_extended_weather_data(coords['lat'], coords['lon'], start_date_str,
                                                               end_date_str)

                    if weather_data:
                        # Calculate soil moisture statistics
                        soil_moisture_top = [entry.get("GWETTOP", 0) for entry in weather_data]
                        soil_moisture_root = [entry.get("GWETROOT", 0) for entry in weather_data]
                        temperatures = [entry.get("T2M", 0) for entry in weather_data]
                        precipitation = [entry.get("PRECTOTCORR", 0) for entry in weather_data]

                        avg_soil_top = sum(soil_moisture_top) / len(soil_moisture_top) if soil_moisture_top else 0
                        avg_soil_root = sum(soil_moisture_root) / len(soil_moisture_root) if soil_moisture_root else 0
                        avg_temp = sum(temperatures) / len(temperatures) if temperatures else 0
                        total_precip = sum(precipitation)

                        st.success(f"Soil moisture analysis complete for {location.title()}!")

                        # Display soil moisture results
                        st.subheader("💧 Soil Moisture Analysis")

                        # Soil moisture status
                        if avg_soil_top > 0.7:
                            st.success("🟢 **SOIL MOISTURE: EXCELLENT**")
                        elif avg_soil_top > 0.5:
                            st.success("🟡 **SOIL MOISTURE: GOOD**")
                        elif avg_soil_top > 0.3:
                            st.warning("🟠 **SOIL MOISTURE: MODERATE**")
                        else:
                            st.error("🔴 **SOIL MOISTURE: LOW**")

                        # Detailed metrics
                        col1, col2, col3, col4 = st.columns(4)

                        with col1:
                            st.metric("Surface Soil Moisture", f"{avg_soil_top:.2f}")
                        with col2:
                            st.metric("Root Zone Moisture", f"{avg_soil_root:.2f}")
                        with col3:
                            st.metric("Avg Temperature", f"{avg_temp:.1f}°C")
                        with col4:
                            st.metric("Total Precipitation", f"{total_precip:.1f} mm")

                        # Soil moisture interpretation
                        st.markdown("**Soil Moisture Interpretation:**")
                        if avg_soil_top > 0.7:
                            st.info(
                                "🌱 Excellent soil moisture conditions. Ideal for most crops and minimal irrigation needed.")
                        elif avg_soil_top > 0.5:
                            st.info("🌾 Good soil moisture levels. Suitable for most agricultural activities.")
                        elif avg_soil_top > 0.3:
                            st.warning(
                                "🌿 Moderate soil moisture. May require supplemental irrigation for optimal crop growth.")
                        else:
                            st.error("🌵 Low soil moisture levels. Irrigation recommended for agricultural activities.")

                        # Visualization
                        with st.expander("📈 Soil Moisture Visualization"):
                            fig = go.Figure()

                            # Add soil moisture traces
                            fig.add_trace(go.Scatter(
                                x=[entry['date'] for entry in weather_data],
                                y=[entry.get('GWETTOP', 0) for entry in weather_data],
                                mode='lines+markers',
                                name='Surface Soil Moisture',
                                line=dict(color='brown')
                            ))

                            fig.add_trace(go.Scatter(
                                x=[entry['date'] for entry in weather_data],
                                y=[entry.get('GWETROOT', 0) for entry in weather_data],
                                mode='lines+markers',
                                name='Root Zone Moisture',
                                line=dict(color='darkgreen')
                            ))

                            # Add optimal range
                            fig.add_hline(y=0.3, line_dash="dash", line_color="red",
                                          annotation_text="Minimum Adequate")
                            fig.add_hline(y=0.7, line_dash="dash", line_color="green",
                                          annotation_text="Optimal Range")

                            fig.update_layout(
                                title="Soil Moisture Trends",
                                xaxis_title="Date",
                                yaxis_title="Soil Moisture (fraction)",
                                height=400
                            )

                            st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.error(f"Sorry, couldn't fetch soil moisture data for {location.title()}.")

            else:
                st.error(
                    "Analysis type not supported yet. Please try weather reports, farming advice, flood risk, site suitability, or drought monitoring.")

    # Quick action buttons
    st.markdown("---")
    st.markdown("### 🚀 Quick Actions")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if st.button("🌊 Flood Risk Demo"):
            st.session_state['demo_query'] = "flood risk assessment for Mumbai this month"

    with col2:
        if st.button("🏞️ Site Suitability Demo"):
            st.session_state['demo_query'] = "site suitability analysis for farming in California"

    with col3:
        if st.button("🌵 Drought Monitor Demo"):
            st.session_state['demo_query'] = "drought conditions in Texas this season"

    with col4:
        if st.button("💧 Soil Moisture Demo"):
            st.session_state['demo_query'] = "soil moisture analysis for Punjab this month"

    # Handle demo queries
    if 'demo_query' in st.session_state:
        st.rerun()

    # Footer
    st.markdown("---")
    st.markdown("*🛰️ Powered by NASA POWER API, Groq AI, and Advanced Agricultural Intelligence*")
    st.markdown(
        "*Features: Weather Analysis • Flood Risk Assessment • Site Suitability • Drought Monitoring • Soil Moisture Analysis*")


if __name__ == "__main__":
    main()