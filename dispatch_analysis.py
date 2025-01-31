import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class DispatchAnalyzer:
    def __init__(self):
        self.shifts_df = None
        self.bookings_df = None
        self.auctions_df = None
        
    def load_data(self, shifts_path, bookings_path, auctions_path):
        """Load and preprocess the data from CSV files"""
        self.shifts_df = pd.read_csv(shifts_path)
        self.bookings_df = pd.read_csv(bookings_path)
        self.auctions_df = pd.read_csv(auctions_path)
        
        # Convert datetime columns
        self.shifts_df['shift_date'] = pd.to_datetime(self.shifts_df['shift_date'])
        self.bookings_df['booked_start_at'] = pd.to_datetime(self.bookings_df['booked_start_at'])
        
    def analyze_supply(self):
        """Analyze supply patterns"""
        # Aggregate shifts by date
        daily_shifts = self.shifts_df.groupby('shift_date').agg({
            'shift_id': 'count',
            'shift_working_hours': 'sum',
            'hourly_rate_eur': 'mean'
        }).reset_index()
        
        return daily_shifts
    
    def analyze_demand(self):
        """Analyze demand patterns"""
        # Aggregate bookings by date
        self.bookings_df['booking_date'] = self.bookings_df['booked_start_at'].dt.date
        daily_bookings = self.bookings_df.groupby('booking_date').agg({
            'booking_uuid': 'count',
            'gross_revenue_eur': 'sum',
            'booked_duration': 'mean',
            'booked_distance': 'mean'
        }).reset_index()
        
        return daily_bookings
    
    def analyze_auctions(self):
        """Analyze auction patterns"""
        auction_metrics = self.auctions_df.agg({
            'auction_corridor_min_price': ['mean', 'std'],
            'auction_corridor_max_price': ['mean', 'std'],
            'auction_winning_price': ['mean', 'std']
        })
        
        # Calculate average profit margin
        self.auctions_df['profit_margin'] = (
            self.auctions_df['auction_winning_price'] - 
            self.auctions_df['auction_corridor_min_price']
        )
        
        return auction_metrics
    
    def calculate_kpis(self):
        """Calculate key performance indicators"""
        kpis = {}
        
        # Supply utilization
        total_hours = self.shifts_df['shift_working_hours'].sum()
        total_bookings = len(self.bookings_df)
        kpis['hours_per_booking'] = total_hours / total_bookings
        
        # Revenue metrics
        total_revenue = self.bookings_df['gross_revenue_eur'].sum()
        total_shift_cost = (
            self.shifts_df['shift_working_hours'] * 
            self.shifts_df['hourly_rate_eur']
        ).sum()
        kpis['revenue_per_hour'] = total_revenue / total_hours
        kpis['cost_per_hour'] = total_shift_cost / total_hours
        
        # Auction efficiency
        avg_auction_margin = self.auctions_df['profit_margin'].mean()
        kpis['avg_auction_margin'] = avg_auction_margin
        
        return kpis
    
    def visualize_patterns(self):
        """Create visualizations for supply-demand patterns"""
        # Time series of supply and demand
        daily_supply = self.analyze_supply()
        daily_demand = self.analyze_demand()
        
        plt.figure(figsize=(12, 6))
        plt.plot(daily_supply['shift_date'], daily_supply['shift_working_hours'], 
                label='Supply (Hours)')
        plt.plot(daily_demand['booking_date'], daily_demand['booking_uuid'], 
                label='Demand (Bookings)')
        plt.title('Supply vs Demand Over Time')
        plt.xlabel('Date')
        plt.ylabel('Count')
        plt.legend()
        plt.tight_layout()
        plt.savefig('supply_demand_pattern.png')
        plt.close()

def main():
    analyzer = DispatchAnalyzer()
    
    # Load data (update paths as needed)
    analyzer.load_data(
        'disp_pre_purchased_shifts.csv',
        'disp_incoming_bookings.csv',
        'historical_auction_data.csv'
    )
    
    # Perform analysis
    supply_analysis = analyzer.analyze_supply()
    demand_analysis = analyzer.analyze_demand()
    auction_analysis = analyzer.analyze_auctions()
    kpis = analyzer.calculate_kpis()
    
    # Generate visualizations
    analyzer.visualize_patterns()
    
    # Print summary
    print("\nSupply Analysis Summary:")
    print(supply_analysis.describe())
    
    print("\nDemand Analysis Summary:")
    print(demand_analysis.describe())
    
    print("\nKPI Summary:")
    for kpi, value in kpis.items():
        print(f"{kpi}: {value:.2f}")

if __name__ == "__main__":
    main()
