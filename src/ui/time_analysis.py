import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# Set page config
st.set_page_config(page_title="Time and Event Analysis", layout="wide")

def load_data():
    """Load all required datasets"""
    data_dir = Path(__file__).parents[2] / "data" / "raw"
    
    # Load main data
    bookings_df = pd.read_csv(data_dir / "bookings.csv")
    auction_df = pd.read_csv(data_dir / "auction_data.csv")
    
    # Convert timestamps
    bookings_df['booking_start_time'] = pd.to_datetime(bookings_df['booking_start_time'])
    bookings_df['booking_end_time'] = pd.to_datetime(bookings_df['booking_end_time'])
    
    # Extract time components
    bookings_df['hour_of_day'] = bookings_df['booking_start_time'].dt.hour
    bookings_df['day_of_week'] = bookings_df['booking_start_time'].dt.day_name()
    bookings_df['date'] = bookings_df['booking_start_time'].dt.date
    
    # Try to load weather and events data
    try:
        weather_df = pd.read_csv(data_dir / "weather_data.csv")
        weather_df['date'] = pd.to_datetime(weather_df['date']).dt.date
    except:
        weather_df = None
        
    try:
        events_df = pd.read_csv(data_dir / "events_data.csv")
        events_df['event_date'] = pd.to_datetime(events_df['event_date']).dt.date
    except:
        events_df = None
    
    return bookings_df, auction_df, weather_df, events_df

def create_time_analysis(merged_df):
    """Create time-based analysis visualizations"""
    st.header("Time-Based Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Hourly analysis
        hourly_stats = merged_df.groupby('hour_of_day')['auction_winning_price'].agg(['mean', 'count']).reset_index()
        
        fig = px.bar(hourly_stats, x='hour_of_day', y='mean',
                    title='Average Auction Price by Hour',
                    labels={'hour_of_day': 'Hour of Day', 'mean': 'Average Price (€)'},
                    text='count')
        fig.update_traces(texttemplate='%{text} bookings', textposition='outside')
        st.plotly_chart(fig)
    
    with col2:
        # Daily analysis
        daily_stats = merged_df.groupby('day_of_week')['auction_winning_price'].mean().reset_index()
        
        fig = px.bar(daily_stats, x='day_of_week', y='auction_winning_price',
                    title='Average Auction Price by Day of Week',
                    labels={'day_of_week': 'Day of Week', 'auction_winning_price': 'Average Price (€)'})
        st.plotly_chart(fig)

def create_weather_analysis(merged_df):
    """Create weather-based analysis visualizations"""
    st.header("Weather Impact Analysis")
    
    if 'weather_condition' not in merged_df.columns:
        st.warning("Weather data not available")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Weather condition analysis
        weather_stats = merged_df.groupby('weather_condition')['auction_winning_price'].agg(['mean', 'count']).reset_index()
        
        fig = px.bar(weather_stats, x='weather_condition', y='mean',
                    title='Average Auction Price by Weather Condition',
                    labels={'weather_condition': 'Weather', 'mean': 'Average Price (€)'},
                    text='count')
        fig.update_traces(texttemplate='%{text} bookings', textposition='outside')
        st.plotly_chart(fig)
    
    with col2:
        # Temperature correlation
        if 'temperature' in merged_df.columns:
            fig = px.scatter(merged_df, x='temperature', y='auction_winning_price',
                           title='Auction Price vs Temperature',
                           labels={'temperature': 'Temperature (°C)', 'auction_winning_price': 'Price (€)'})
            st.plotly_chart(fig)

def create_event_analysis(merged_df):
    """Create event-based analysis visualizations"""
    st.header("Event Impact Analysis")
    
    if 'event_type' not in merged_df.columns:
        st.warning("Event data not available")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Event type analysis
        event_stats = merged_df.groupby('event_type')['auction_winning_price'].agg(['mean', 'count']).reset_index()
        
        fig = px.bar(event_stats, x='event_type', y='mean',
                    title='Average Auction Price by Event Type',
                    labels={'event_type': 'Event Type', 'mean': 'Average Price (€)'},
                    text='count')
        fig.update_traces(texttemplate='%{text} bookings', textposition='outside')
        st.plotly_chart(fig)
    
    with col2:
        # Event vs Non-event days
        merged_df['has_event'] = merged_df['event_type'].notna().map({True: 'Event Day', False: 'Regular Day'})
        event_day_stats = merged_df.groupby('has_event')['auction_winning_price'].agg(['mean', 'count']).reset_index()
        
        fig = px.bar(event_day_stats, x='has_event', y='mean',
                    title='Average Auction Price: Event vs Regular Days',
                    labels={'has_event': 'Day Type', 'mean': 'Average Price (€)'},
                    text='count')
        fig.update_traces(texttemplate='%{text} bookings', textposition='outside')
        st.plotly_chart(fig)

def main():
    st.title("Time, Weather, and Event Analysis Dashboard")
    
    try:
        # Load all data
        bookings_df, auction_df, weather_df, events_df = load_data()
        
        # Merge bookings with auction data
        merged_df = bookings_df.merge(auction_df, on='booking_uuid', how='inner')
        
        # Merge with weather and events if available
        if weather_df is not None:
            merged_df = merged_df.merge(weather_df, on='date', how='left')
        
        if events_df is not None:
            merged_df = merged_df.merge(events_df, left_on='date', right_on='event_date', how='left')
        
        # Create analyses
        create_time_analysis(merged_df)
        create_weather_analysis(merged_df)
        create_event_analysis(merged_df)
        
        # Correlation Analysis
        st.header("Correlation Analysis")
        
        # Select numerical columns
        numeric_cols = merged_df.select_dtypes(include=[np.number]).columns
        correlation = merged_df[numeric_cols].corr()
        
        fig = px.imshow(correlation,
                       labels=dict(color="Correlation"),
                       x=correlation.columns,
                       y=correlation.columns,
                       color_continuous_scale='RdBu')
        fig.update_traces(text=correlation.round(2), texttemplate='%{text}')
        st.plotly_chart(fig)
        
    except Exception as e:
        st.error(f"Error in analysis: {str(e)}")
        st.write("Please ensure all required data files are present and properly formatted.")

if __name__ == "__main__":
    main()
