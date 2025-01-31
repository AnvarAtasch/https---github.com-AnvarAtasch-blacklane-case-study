import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np

# Set page config
st.set_page_config(
    page_title="Blacklane Dispatching - Seasonality Analysis",
    page_icon="🚗",
    layout="wide"
)

# Add custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stPlotlyChart {
        background-color: white;
        border-radius: 5px;
        padding: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    h1, h2, h3 {
        color: #1E3D59;
    }
    .metric-card {
        background-color: white;
        padding: 1rem;
        border-radius: 5px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
    </style>
    """, unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load and preprocess the data"""
    bookings_df = pd.read_csv('../../data/raw/disp_incoming_bookings.csv')
    bookings_df['booked_start_at'] = pd.to_datetime(bookings_df['booked_start_at'])
    return bookings_df

def create_daily_pattern(df):
    """Create daily pattern visualization"""
    df['day_of_week'] = df['booked_start_at'].dt.day_name()
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    daily_counts = df['day_of_week'].value_counts().reindex(day_order)
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=daily_counts.index,
        y=daily_counts.values,
        marker_color='#1E88E5',
        hovertemplate='Day: %{x}<br>Bookings: %{y}<extra></extra>'
    ))
    
    fig.update_layout(
        title='Booking Demand by Day of Week',
        xaxis_title='Day of Week',
        yaxis_title='Number of Bookings',
        template='plotly_white',
        height=500
    )
    
    return fig, daily_counts

def create_hourly_pattern(df):
    """Create hourly pattern visualization"""
    df['hour_of_day'] = df['booked_start_at'].dt.hour
    hourly_counts = df['hour_of_day'].value_counts().sort_index()
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=hourly_counts.index,
        y=hourly_counts.values,
        mode='lines+markers',
        marker=dict(size=8, color='#43A047'),
        line=dict(width=2, color='#43A047'),
        hovertemplate='Hour: %{x}:00<br>Bookings: %{y}<extra></extra>'
    ))
    
    fig.update_layout(
        title='Booking Demand by Hour of Day',
        xaxis_title='Hour of Day',
        yaxis_title='Number of Bookings',
        xaxis=dict(tickmode='array', ticktext=[f'{i:02d}:00' for i in range(24)], tickvals=list(range(24))),
        template='plotly_white',
        height=500
    )
    
    return fig, hourly_counts

def create_heatmap(df):
    """Create day-hour heatmap"""
    df['day_of_week'] = df['booked_start_at'].dt.day_name()
    df['hour_of_day'] = df['booked_start_at'].dt.hour
    
    # Create pivot table for heatmap
    heatmap_data = pd.pivot_table(
        df,
        values='booking_uuid',
        index='day_of_week',
        columns='hour_of_day',
        aggfunc='count',
        fill_value=0
    )
    
    # Reorder days
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    heatmap_data = heatmap_data.reindex(day_order)
    
    fig = go.Figure(data=go.Heatmap(
        z=heatmap_data.values,
        x=[f'{i:02d}:00' for i in range(24)],
        y=heatmap_data.index,
        colorscale='Viridis',
        hoverongaps=False,
        hovertemplate='Day: %{y}<br>Hour: %{x}<br>Bookings: %{z}<extra></extra>'
    ))
    
    fig.update_layout(
        title='Booking Demand Heatmap (Day vs Hour)',
        xaxis_title='Hour of Day',
        yaxis_title='Day of Week',
        template='plotly_white',
        height=500
    )
    
    return fig

def main():
    st.title("🚗 Blacklane Dispatching - Seasonality Analysis")
    st.write("Analyze booking patterns across different time periods")
    
    try:
        # Load data
        with st.spinner('Loading data...'):
            df = load_data()
        
        # Date range filter
        st.sidebar.header("Filters")
        min_date = df['booked_start_at'].min().date()
        max_date = df['booked_start_at'].max().date()
        
        date_range = st.sidebar.date_input(
            "Select Date Range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date
        )
        
        if len(date_range) == 2:
            start_date, end_date = date_range
            mask = (df['booked_start_at'].dt.date >= start_date) & (df['booked_start_at'].dt.date <= end_date)
            filtered_df = df[mask]
        else:
            filtered_df = df
        
        # Display metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(
                "Total Bookings",
                f"{len(filtered_df):,}",
                help="Total number of bookings in selected period"
            )
        with col2:
            avg_daily = len(filtered_df) / len(filtered_df['booked_start_at'].dt.date.unique())
            st.metric(
                "Average Daily Bookings",
                f"{avg_daily:.1f}",
                help="Average number of bookings per day"
            )
        with col3:
            peak_hour = filtered_df['booked_start_at'].dt.hour.mode().iloc[0]
            st.metric(
                "Peak Hour",
                f"{peak_hour:02d}:00",
                help="Most common booking hour"
            )
        
        # Create visualizations
        tab1, tab2, tab3 = st.tabs(["Daily Pattern", "Hourly Pattern", "Heatmap"])
        
        with tab1:
            daily_fig, daily_counts = create_daily_pattern(filtered_df)
            st.plotly_chart(daily_fig, use_container_width=True)
            
            # Add insights
            busiest_day = daily_counts.idxmax()
            quietest_day = daily_counts.idxmin()
            st.info(f"""
                📊 **Key Insights:**
                - Busiest day: **{busiest_day}** ({daily_counts[busiest_day]:,} bookings)
                - Quietest day: **{quietest_day}** ({daily_counts[quietest_day]:,} bookings)
                - Weekend vs Weekday ratio: **{daily_counts[['Saturday', 'Sunday']].mean() / daily_counts[['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']].mean():.2f}**
            """)
        
        with tab2:
            hourly_fig, hourly_counts = create_hourly_pattern(filtered_df)
            st.plotly_chart(hourly_fig, use_container_width=True)
            
            # Add insights
            peak_hours = hourly_counts.nlargest(3)
            st.info(f"""
                📊 **Key Insights:**
                - Peak hours:
                  1. **{peak_hours.index[0]:02d}:00** ({peak_hours.iloc[0]:,} bookings)
                  2. **{peak_hours.index[1]:02d}:00** ({peak_hours.iloc[1]:,} bookings)
                  3. **{peak_hours.index[2]:02d}:00** ({peak_hours.iloc[2]:,} bookings)
                - Off-peak hours (lowest demand): **{hourly_counts.nsmallest(3).index.tolist()}:00**
            """)
        
        with tab3:
            heatmap_fig = create_heatmap(filtered_df)
            st.plotly_chart(heatmap_fig, use_container_width=True)
            st.info("""
                🔍 **How to read the heatmap:**
                - Darker colors indicate higher booking volumes
                - X-axis shows hours of the day
                - Y-axis shows days of the week
                - Hover over cells to see exact booking counts
            """)
    
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.error("Please make sure the data files are in the correct location: data/raw/")

if __name__ == "__main__":
    main()
