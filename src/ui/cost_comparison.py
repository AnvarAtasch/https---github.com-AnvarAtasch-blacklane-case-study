import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# Set page config
st.set_page_config(page_title="Cost Comparison Analysis", layout="wide")

# Function to load data
def load_data():
    data_dir = Path(__file__).parents[2] / "data" / "raw"
    shifts_df = pd.read_csv(data_dir / "pre_purchased_shifts.csv")
    bookings_df = pd.read_csv(data_dir / "bookings.csv")
    auction_df = pd.read_csv(data_dir / "auction_data.csv")
    return shifts_df, bookings_df, auction_df

def calculate_costs(shifts_df, bookings_df, auction_df):
    # Calculate total cost for pre-purchased shifts
    shifts_df['total_shift_cost'] = shifts_df['shift_working_hours'] * shifts_df['hourly_rate_eur']
    
    # Merge bookings with auction data
    bookings_with_auction = bookings_df.merge(auction_df, on='booking_uuid', how='left')
    
    # Merge with shift data
    bookings_with_shift = bookings_with_auction.merge(
        shifts_df[['shift_id', 'chauffeur_uuid', 'hourly_rate_eur']], 
        on='chauffeur_uuid', 
        how='left'
    )
    
    # Calculate costs
    bookings_with_shift['pre_purchased_cost'] = bookings_with_shift['hourly_rate_eur'] * bookings_with_shift['booked_duration']
    bookings_with_shift['cost_diff'] = bookings_with_shift['auction_winning_price'] - bookings_with_shift['pre_purchased_cost']
    
    return bookings_with_shift

def main():
    st.title("Cost Comparison Analysis: Pre-Purchased Shifts vs. Auction")
    
    # Load data
    try:
        shifts_df, bookings_df, auction_df = load_data()
        bookings_with_costs = calculate_costs(shifts_df, bookings_df, auction_df)
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return

    # Summary metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        total_shift_cost = shifts_df['total_shift_cost'].sum()
        st.metric("Total Pre-Purchased Shifts Cost", f"€{total_shift_cost:,.2f}")
    
    with col2:
        avg_auction_price = auction_df['auction_winning_price'].mean()
        st.metric("Average Auction Price", f"€{avg_auction_price:,.2f}")
    
    with col3:
        avg_cost_diff = bookings_with_costs['cost_diff'].mean()
        st.metric("Average Cost Difference", f"€{avg_cost_diff:,.2f}")

    # Cost Distribution Analysis
    st.header("Cost Distribution Analysis")
    
    # Create tabs for different visualizations
    tab1, tab2, tab3 = st.tabs(["Cost Difference Distribution", "Cost Comparison by Duration", "Time-based Analysis"])
    
    with tab1:
        # Distribution plot of cost differences
        fig = px.histogram(
            bookings_with_costs,
            x='cost_diff',
            nbins=50,
            title='Distribution of Cost Differences (Auction - Pre-Purchased)',
            labels={'cost_diff': 'Cost Difference (€)'}
        )
        fig.add_vline(x=0, line_dash="dash", line_color="red")
        st.plotly_chart(fig)
        
        # Calculate and display percentages
        total_bookings = len(bookings_with_costs['cost_diff'].dropna())
        auction_cheaper = (bookings_with_costs['cost_diff'] < 0).sum()
        pre_purchased_cheaper = (bookings_with_costs['cost_diff'] > 0).sum()
        
        st.write(f"Auction cheaper in {auction_cheaper/total_bookings*100:.1f}% of cases")
        st.write(f"Pre-purchased cheaper in {pre_purchased_cheaper/total_bookings*100:.1f}% of cases")

    with tab2:
        # Scatter plot of costs vs duration
        fig = px.scatter(
            bookings_with_costs,
            x='booked_duration',
            y=['auction_winning_price', 'pre_purchased_cost'],
            title='Cost vs. Booking Duration',
            labels={
                'booked_duration': 'Booking Duration',
                'value': 'Cost (€)',
                'variable': 'Cost Type'
            }
        )
        st.plotly_chart(fig)

    with tab3:
        # Time-based analysis
        bookings_with_costs['hour'] = pd.to_datetime(bookings_with_costs['booking_start_time']).dt.hour
        hourly_cost_diff = bookings_with_costs.groupby('hour')['cost_diff'].mean().reset_index()
        
        fig = px.line(
            hourly_cost_diff,
            x='hour',
            y='cost_diff',
            title='Average Cost Difference by Hour',
            labels={
                'hour': 'Hour of Day',
                'cost_diff': 'Average Cost Difference (€)'
            }
        )
        fig.add_hline(y=0, line_dash="dash", line_color="red")
        st.plotly_chart(fig)

    # Additional Analysis
    st.header("Cost Effectiveness Thresholds")
    
    # Calculate cost effectiveness by different factors
    try:
        duration_bins = pd.qcut(bookings_with_costs['booked_duration'], q=5, duplicates='drop')
        cost_by_duration = bookings_with_costs.groupby(duration_bins)['cost_diff'].mean()
        
        st.write("Average cost difference by booking duration ranges:")
        cost_summary = pd.DataFrame({
            'Duration Range': cost_by_duration.index.astype(str),
            'Average Cost Difference (€)': cost_by_duration.values.round(2)
        })
        st.dataframe(cost_summary)
        
        # Add a bar chart visualization
        fig = px.bar(
            cost_summary,
            x='Duration Range',
            y='Average Cost Difference (€)',
            title='Average Cost Difference by Booking Duration',
            labels={'Duration Range': 'Booking Duration Range', 'Average Cost Difference (€)': 'Cost Difference (€)'}
        )
        st.plotly_chart(fig)
        
    except Exception as e:
        st.warning(f"Note: Could not generate duration-based analysis due to insufficient unique duration values in the dataset. Please ensure you have enough diverse booking durations for this analysis.")
        st.write("Consider analyzing the raw cost differences instead:")
        basic_stats = bookings_with_costs['cost_diff'].describe()
        st.dataframe(basic_stats)

if __name__ == "__main__":
    main()
