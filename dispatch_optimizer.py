import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime, timedelta

class DispatchOptimizer:
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        
    def prepare_features(self, booking, available_shifts, historical_auctions):
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
            
        # Historical auction features
        relevant_auctions = historical_auctions[
            (historical_auctions['booked_duration'].between(
                booking['booked_duration'] * 0.8,
                booking['booked_duration'] * 1.2
            ))
        ]
        
        if len(relevant_auctions) > 0:
            features['avg_auction_price'] = relevant_auctions['auction_winning_price'].mean()
            features['auction_success_rate'] = (
                relevant_auctions['auction_winning_price'].notna().mean()
            )
        else:
            features['avg_auction_price'] = 0
            features['auction_success_rate'] = 0
            
        return pd.Series(features)
    
    def train_model(self, historical_data):
        """Train the decision model using historical data"""
        X = pd.DataFrame([
            self.prepare_features(row, row['available_shifts'], row['historical_auctions'])
            for _, row in historical_data.iterrows()
        ])
        
        # Target: 1 if shift assignment was profitable, 0 if auction was better
        y = historical_data['assignment_profitable']
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train model
        self.model.fit(X_scaled, y)
        
    def make_decision(self, booking, available_shifts, historical_auctions):
        """Decide whether to assign to pre-purchased shift or send to auction"""
        features = self.prepare_features(booking, available_shifts, historical_auctions)
        features_scaled = self.scaler.transform(features.values.reshape(1, -1))
        
        # Get probability of shift assignment being profitable
        prob_profitable = self.model.predict_proba(features_scaled)[0][1]
        
        decision = {
            'assign_to_shift': prob_profitable > 0.5,
            'confidence': prob_profitable if prob_profitable > 0.5 else 1 - prob_profitable,
            'expected_profit': self.calculate_expected_profit(
                booking, available_shifts, historical_auctions, prob_profitable
            )
        }
        
        return decision
    
    def calculate_expected_profit(self, booking, available_shifts, historical_auctions, prob_profitable):
        """Calculate expected profit for the decision"""
        revenue = booking['gross_revenue_eur']
        
        # Calculate shift cost
        if len(available_shifts) > 0:
            shift_cost = (
                booking['booked_duration'] * 
                available_shifts['hourly_rate_eur'].mean()
            )
        else:
            shift_cost = float('inf')
            
        # Calculate expected auction cost
        relevant_auctions = historical_auctions[
            (historical_auctions['booked_duration'].between(
                booking['booked_duration'] * 0.8,
                booking['booked_duration'] * 1.2
            ))
        ]
        
        if len(relevant_auctions) > 0:
            expected_auction_price = relevant_auctions['auction_winning_price'].mean()
        else:
            expected_auction_price = revenue * 0.7  # Conservative estimate
            
        # Calculate expected profit for each option
        shift_profit = revenue - shift_cost
        auction_profit = revenue - expected_auction_price
        
        # Return weighted average based on probability
        return prob_profitable * shift_profit + (1 - prob_profitable) * auction_profit

def main():
    # Example usage
    optimizer = DispatchOptimizer()
    
    # Load historical data and train model
    # This would need to be replaced with actual historical data
    historical_data = pd.DataFrame({
        'booking_id': range(100),
        'assignment_profitable': np.random.choice([0, 1], size=100),
        # Add other necessary columns
    })
    
    optimizer.train_model(historical_data)
    
    # Make decision for new booking
    booking = {
        'booked_duration': 120,  # minutes
        'booked_distance': 30,   # km
        'gross_revenue_eur': 100,
        'booked_start_at': datetime.now()
    }
    
    available_shifts = pd.DataFrame({
        'hourly_rate_eur': [25, 30, 28]
    })
    
    historical_auctions = pd.DataFrame({
        'booked_duration': [110, 120, 130],
        'auction_winning_price': [80, 85, 90]
    })
    
    decision = optimizer.make_decision(booking, available_shifts, historical_auctions)
    print("Decision:", decision)

if __name__ == "__main__":
    main()
