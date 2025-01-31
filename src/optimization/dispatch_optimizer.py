import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime, timedelta
from pathlib import Path

class DispatchOptimizer:
    def __init__(self, data_dir='../../data'):
        self.data_dir = Path(data_dir)
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
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
        
    def prepare_features(self, booking, available_shifts):
        """Prepare features for the decision model"""
        features = {}
        
        # Booking features
        features['booked_duration'] = booking['booked_duration']
        features['booked_distance'] = booking['booked_distance']
        features['expected_revenue'] = booking['gross_revenue_eur']
        
        # Time-based features
        booking_time = pd.to_datetime(booking['booked_start_at'])
        features['hour_of_day'] = booking_time.hour
        features['day_of_week'] = booking_time.dayofweek
        
        # Supply features
        features['available_shifts'] = len(available_shifts)
        if len(available_shifts) > 0:
            features['avg_shift_cost'] = available_shifts['hourly_rate_eur'].mean()
        else:
            features['avg_shift_cost'] = 0
            
        # Historical auction features for similar bookings
        similar_auctions = self.get_similar_auctions(booking)
        if len(similar_auctions) > 0:
            features['avg_auction_price'] = similar_auctions['auction_winning_price'].mean()
            features['auction_success_rate'] = similar_auctions['auction_winning_price'].notna().mean()
        else:
            features['avg_auction_price'] = 0
            features['auction_success_rate'] = 0
            
        return pd.Series(features)
    
    def get_similar_auctions(self, booking):
        """Get historical auctions for similar bookings"""
        duration_lower = booking['booked_duration'] * 0.8
        duration_upper = booking['booked_duration'] * 1.2
        
        return self.auctions_df[
            self.auctions_df['booking_uuid'].isin(
                self.bookings_df[
                    (self.bookings_df['booked_duration'].between(duration_lower, duration_upper)) &
                    (self.bookings_df['booked_distance'].between(
                        booking['booked_distance'] * 0.8,
                        booking['booked_distance'] * 1.2
                    ))
                ]['booking_uuid']
            )
        ]
    
    def get_available_shifts(self, booking_time):
        """Get available shifts for a given booking time"""
        booking_date = booking_time.date()
        return self.shifts_df[
            (self.shifts_df['shift_date'].dt.date == booking_date)
        ]
    
    def train_model(self):
        """Train the decision model using historical data"""
        print("Preparing training data...")
        
        # Prepare training data
        training_data = []
        for _, booking in self.bookings_df.iterrows():
            available_shifts = self.get_available_shifts(booking['booked_start_at'])
            features = self.prepare_features(booking, available_shifts)
            
            # Determine if shift assignment was profitable
            if booking['booking_uuid'] in self.auctions_df['booking_uuid'].values:
                auction_price = self.auctions_df[
                    self.auctions_df['booking_uuid'] == booking['booking_uuid']
                ]['auction_winning_price'].iloc[0]
                
                shift_cost = (
                    booking['booked_duration'] / 60 * 
                    available_shifts['hourly_rate_eur'].mean()
                    if len(available_shifts) > 0 else float('inf')
                )
                
                # Target: 1 if shift assignment would have been more profitable
                profitable = shift_cost < auction_price
                
                training_data.append((features, profitable))
        
        # Convert to DataFrame
        X = pd.DataFrame([data[0] for data in training_data])
        y = np.array([data[1] for data in training_data])
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train model
        print("Training model...")
        self.model.fit(X_scaled, y)
        print("Model training completed")
        
    def make_decision(self, booking):
        """Decide whether to assign to pre-purchased shift or send to auction"""
        # Get available shifts
        available_shifts = self.get_available_shifts(booking['booked_start_at'])
        
        # Prepare features
        features = self.prepare_features(booking, available_shifts)
        features_scaled = self.scaler.transform(features.values.reshape(1, -1))
        
        # Get probability of shift assignment being profitable
        prob_profitable = self.model.predict_proba(features_scaled)[0][1]
        
        # Calculate expected profits
        shift_cost = (
            booking['booked_duration'] / 60 * 
            available_shifts['hourly_rate_eur'].mean()
            if len(available_shifts) > 0 else float('inf')
        )
        
        similar_auctions = self.get_similar_auctions(booking)
        expected_auction_price = (
            similar_auctions['auction_winning_price'].mean()
            if len(similar_auctions) > 0
            else booking['gross_revenue_eur'] * 0.7
        )
        
        decision = {
            'assign_to_shift': prob_profitable > 0.5 and len(available_shifts) > 0,
            'confidence': prob_profitable if prob_profitable > 0.5 else 1 - prob_profitable,
            'expected_shift_cost': shift_cost,
            'expected_auction_price': expected_auction_price,
            'expected_profit': booking['gross_revenue_eur'] - min(shift_cost, expected_auction_price)
        }
        
        return decision

def main():
    # Initialize optimizer
    optimizer = DispatchOptimizer()
    
    # Load data
    print("Loading data...")
    optimizer.load_data()
    
    # Train model
    optimizer.train_model()
    
    # Example decision
    example_booking = {
        'booked_duration': 120,  # minutes
        'booked_distance': 30,   # km
        'gross_revenue_eur': 100,
        'booked_start_at': pd.Timestamp.now()
    }
    
    decision = optimizer.make_decision(example_booking)
    
    print("\nExample Decision:")
    print(f"Assign to shift: {decision['assign_to_shift']}")
    print(f"Confidence: {decision['confidence']:.2f}")
    print(f"Expected profit: €{decision['expected_profit']:.2f}")
    print(f"Expected shift cost: €{decision['expected_shift_cost']:.2f}")
    print(f"Expected auction price: €{decision['expected_auction_price']:.2f}")

if __name__ == "__main__":
    main()
