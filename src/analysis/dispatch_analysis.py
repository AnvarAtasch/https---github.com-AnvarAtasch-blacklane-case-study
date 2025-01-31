import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from pathlib import Path

class DispatchAnalyzer:
    def __init__(self, data_dir='../../data'):
        self.data_dir = Path(data_dir)
        self.shifts_df = None
        self.bookings_df = None
        self.auctions_df = None
        
    def load_data(self):
        """Load data from CSV files"""
        self.shifts_df = pd.read_csv(
            self.data_dir / 'raw/disp_pre_purchased_shifts.csv'
        )
        self.bookings_df = pd.read_csv(
            self.data_dir / 'raw/disp_incoming_bookings.csv'
        )
        self.auctions_df = pd.read_csv(
            self.data_dir / 'raw/disp_historical_auction_data.csv'
        )
        
        # Convert datetime columns
        self.shifts_df['shift_date'] = pd.to_datetime(self.shifts_df['shift_date'])
        self.bookings_df['booked_start_at'] = pd.to_datetime(self.bookings_df['booked_start_at'])
        
        print("Data loaded successfully:")
        print(f"Shifts: {len(self.shifts_df)} records")
        print(f"Bookings: {len(self.bookings_df)} records")
        print(f"Auctions: {len(self.auctions_df)} records")
        
    def analyze_supply(self):
        """Analyze supply patterns"""
        supply_analysis = {
            'daily_shifts': self.shifts_df.groupby('shift_date').agg({
                'shift_id': 'count',
                'shift_working_hours': 'sum',
                'hourly_rate_eur': ['mean', 'std']
            }),
            'total_hours': self.shifts_df['shift_working_hours'].sum(),
            'avg_hourly_rate': self.shifts_df['hourly_rate_eur'].mean(),
            'unique_chauffeurs': self.shifts_df['chauffeur_uuid'].nunique()
        }
        
        # Save analysis results
        supply_analysis['daily_shifts'].to_csv(
            self.data_dir / 'processed/supply_analysis.csv'
        )
        
        return supply_analysis
    
    def analyze_demand(self):
        """Analyze demand patterns"""
        self.bookings_df['booking_date'] = self.bookings_df['booked_start_at'].dt.date
        
        demand_analysis = {
            'daily_bookings': self.bookings_df.groupby('booking_date').agg({
                'booking_uuid': 'count',
                'gross_revenue_eur': ['sum', 'mean'],
                'booked_duration': ['mean', 'std'],
                'booked_distance': ['mean', 'std']
            }),
            'total_revenue': self.bookings_df['gross_revenue_eur'].sum(),
            'avg_trip_duration': self.bookings_df['booked_duration'].mean(),
            'avg_trip_distance': self.bookings_df['booked_distance'].mean()
        }
        
        # Save analysis results
        demand_analysis['daily_bookings'].to_csv(
            self.data_dir / 'processed/demand_analysis.csv'
        )
        
        return demand_analysis
    
    def analyze_auctions(self):
        """Analyze auction patterns"""
        auction_analysis = {
            'price_metrics': self.auctions_df.agg({
                'auction_corridor_min_price': ['mean', 'std', 'min', 'max'],
                'auction_corridor_max_price': ['mean', 'std', 'min', 'max'],
                'auction_winning_price': ['mean', 'std', 'min', 'max']
            }),
            'margin_analysis': pd.DataFrame({
                'margin': (
                    self.auctions_df['auction_winning_price'] -
                    self.auctions_df['auction_corridor_min_price']
                )
            }).agg(['mean', 'std', 'min', 'max'])
        }
        
        # Save analysis results
        pd.DataFrame(auction_analysis['price_metrics']).to_csv(
            self.data_dir / 'processed/auction_analysis.csv'
        )
        
        return auction_analysis
    
    def calculate_kpis(self):
        """Calculate key performance indicators"""
        total_hours = self.shifts_df['shift_working_hours'].sum()
        total_bookings = len(self.bookings_df)
        total_revenue = self.bookings_df['gross_revenue_eur'].sum()
        total_shift_cost = (
            self.shifts_df['shift_working_hours'] * 
            self.shifts_df['hourly_rate_eur']
        ).sum()
        
        kpis = {
            'hours_per_booking': total_hours / total_bookings,
            'revenue_per_hour': total_revenue / total_hours,
            'cost_per_hour': total_shift_cost / total_hours,
            'gross_margin': (total_revenue - total_shift_cost) / total_revenue,
            'auction_success_rate': self.auctions_df['auction_winning_price'].notna().mean()
        }
        
        # Save KPIs
        pd.DataFrame(kpis, index=[0]).to_csv(
            self.data_dir / 'processed/kpis.csv'
        )
        
        return kpis
    
    def visualize_patterns(self):
        """Create visualizations for supply-demand patterns"""
        # Create output directory
        output_dir = self.data_dir / 'processed/visualizations'
        output_dir.mkdir(exist_ok=True)
        
        # 1. Supply vs Demand Over Time
        plt.figure(figsize=(12, 6))
        daily_supply = self.shifts_df.groupby('shift_date')['shift_working_hours'].sum()
        daily_demand = self.bookings_df.groupby('booked_start_at').size()
        
        plt.plot(daily_supply.index, daily_supply.values, label='Supply (Hours)')
        plt.plot(daily_demand.index, daily_demand.values, label='Demand (Bookings)')
        plt.title('Supply vs Demand Over Time')
        plt.xlabel('Date')
        plt.ylabel('Count')
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / 'supply_demand_pattern.png')
        plt.close()
        
        # 2. Auction Price Distribution
        plt.figure(figsize=(10, 6))
        sns.histplot(data=self.auctions_df, x='auction_winning_price', bins=50)
        plt.title('Distribution of Auction Winning Prices')
        plt.xlabel('Price (EUR)')
        plt.ylabel('Count')
        plt.tight_layout()
        plt.savefig(output_dir / 'auction_price_distribution.png')
        plt.close()
        
        # 3. Hourly Rate Distribution
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=self.shifts_df, y='hourly_rate_eur')
        plt.title('Distribution of Hourly Rates')
        plt.ylabel('Rate (EUR)')
        plt.tight_layout()
        plt.savefig(output_dir / 'hourly_rate_distribution.png')
        plt.close()

def main():
    # Initialize analyzer
    analyzer = DispatchAnalyzer()
    
    # Load data
    analyzer.load_data()
    
    # Perform analysis
    supply_analysis = analyzer.analyze_supply()
    demand_analysis = analyzer.analyze_demand()
    auction_analysis = analyzer.analyze_auctions()
    kpis = analyzer.calculate_kpis()
    
    # Generate visualizations
    analyzer.visualize_patterns()
    
    # Print summary
    print("\nAnalysis Summary:")
    print("\nSupply Metrics:")
    print(f"Total Hours: {supply_analysis['total_hours']:.2f}")
    print(f"Average Hourly Rate: €{supply_analysis['avg_hourly_rate']:.2f}")
    
    print("\nDemand Metrics:")
    print(f"Total Revenue: €{demand_analysis['total_revenue']:.2f}")
    print(f"Average Trip Duration: {demand_analysis['avg_trip_duration']:.2f} minutes")
    
    print("\nKPIs:")
    for kpi, value in kpis.items():
        print(f"{kpi}: {value:.2f}")

if __name__ == "__main__":
    main()
