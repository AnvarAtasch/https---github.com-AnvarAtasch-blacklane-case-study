# Blacklane Case Study

## Overview
This repository contains a comprehensive analysis of taxi service data, focusing on cost comparison between pre-purchased shifts and auction participation.

## Dashboards
1. Cost Comparison Dashboard (`src/ui/cost_comparison.py`)
   - Analyzes cost differences between pre-purchased shifts and auction prices
   - Visualizes cost distribution and trends

2. Geographical Analysis Dashboard (`src/ui/geographical_analysis.py`)
   - Maps pickup and dropoff locations
   - Analyzes price variations by area
   - Identifies price hotspots

3. Time Analysis Dashboard (`src/ui/time_analysis.py`)
   - Examines price variations by time of day and day of week
   - Analyzes impact of weather and events on prices
   - Provides correlation analysis

## Data
Sample data files are located in `data/raw/`:
- `bookings.csv`: Booking information
- `auction_data.csv`: Auction results
- `pre_purchased_shifts.csv`: Pre-purchased shift data
- `weather_data.csv`: Weather conditions
- `events_data.csv`: Special events data

## Setup
1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Unix/macOS
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run dashboards:
```bash
streamlit run src/ui/cost_comparison.py
streamlit run src/ui/geographical_analysis.py
streamlit run src/ui/time_analysis.py
```

## Analysis Results
The analysis covers three main hypotheses:
1. Cost Effectiveness of Pre-purchased Shifts vs Auctions
2. Impact of Booking Duration on Cost Differences
3. Geographical Influence on Auction Prices
4. Time and Event Impact on Auction Dynamics
