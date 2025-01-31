import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import folium
from streamlit_folium import st_folium
from sklearn.cluster import KMeans

# Set page config
st.set_page_config(page_title="Geographical Analysis", layout="wide")

def load_data():
    data_dir = Path(__file__).parents[2] / "data" / "raw"
    bookings_df = pd.read_csv(data_dir / "bookings.csv")
    auction_df = pd.read_csv(data_dir / "auction_data.csv")
    return bookings_df, auction_df

def create_grid(df, grid_size=0.01):
    """Create grid coordinates for geographical analysis"""
    df['pickup_grid_x'] = (df['pickup_longitude'] / grid_size).astype(int)
    df['pickup_grid_y'] = (df['pickup_latitude'] / grid_size).astype(int)
    df['dropoff_grid_x'] = (df['dropoff_longitude'] / grid_size).astype(int)
    df['dropoff_grid_y'] = (df['dropoff_latitude'] / grid_size).astype(int)
    return df

def create_heatmap(data, lat_col, lon_col, value_col, title):
    """Create a heatmap using plotly"""
    fig = px.density_mapbox(
        data,
        lat=lat_col,
        lon=lon_col,
        z=value_col,
        radius=30,
        center=dict(lat=data[lat_col].mean(), lon=data[lon_col].mean()),
        zoom=10,
        mapbox_style="stamen-terrain",
        title=title
    )
    return fig

def perform_clustering(data, lat_col, lon_col, n_clusters=5):
    """Perform K-means clustering on geographical points"""
    coords = np.column_stack([data[lon_col], data[lat_col]])
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    data['cluster'] = kmeans.fit_predict(coords)
    return data

def main():
    st.title("Geographical Analysis: Impact on Auction Prices")
    
    try:
        # Load and merge data
        bookings_df, auction_df = load_data()
        merged_df = bookings_df.merge(auction_df, on='booking_uuid', how='inner')
        
        # Add sample coordinates if not present
        if 'pickup_latitude' not in merged_df.columns:
            st.warning("Sample coordinates being used for demonstration")
            # Generate sample coordinates around a city center
            center_lat, center_lon = 52.5200, 13.4050  # Example: Berlin
            n_rows = len(merged_df)
            merged_df['pickup_latitude'] = center_lat + np.random.normal(0, 0.02, n_rows)
            merged_df['pickup_longitude'] = center_lon + np.random.normal(0, 0.02, n_rows)
            merged_df['dropoff_latitude'] = center_lat + np.random.normal(0, 0.02, n_rows)
            merged_df['dropoff_longitude'] = center_lon + np.random.normal(0, 0.02, n_rows)
        
        # Create analysis tabs
        tab1, tab2, tab3 = st.tabs(["Price Heatmaps", "Cluster Analysis", "Grid Analysis"])
        
        with tab1:
            st.header("Price Distribution Heatmaps")
            
            # Pickup locations heatmap
            pickup_fig = create_heatmap(
                merged_df,
                'pickup_latitude',
                'pickup_longitude',
                'auction_winning_price',
                'Auction Prices by Pickup Location'
            )
            st.plotly_chart(pickup_fig)
            
            # Dropoff locations heatmap
            dropoff_fig = create_heatmap(
                merged_df,
                'dropoff_latitude',
                'dropoff_longitude',
                'auction_winning_price',
                'Auction Prices by Dropoff Location'
            )
            st.plotly_chart(dropoff_fig)
        
        with tab2:
            st.header("Cluster Analysis")
            
            # Calculate max possible clusters based on data size
            max_clusters = min(10, len(merged_df))
            default_clusters = min(3, max_clusters)
            
            if max_clusters < 2:
                st.warning("Not enough data points for clustering analysis. Need at least 2 data points.")
            else:
                n_clusters = st.slider("Number of clusters", 2, max_clusters, default_clusters)
                
                # Perform clustering on pickup locations
                clustered_df = perform_clustering(
                    merged_df,
                    'pickup_latitude',
                    'pickup_longitude',
                    n_clusters
                )
                
                # Calculate average prices per cluster
                cluster_stats = clustered_df.groupby('cluster').agg({
                    'auction_winning_price': ['mean', 'count', 'std']
                }).round(2)
                
                st.write("Cluster Statistics:")
                st.dataframe(cluster_stats)
                
                # Visualize clusters
                cluster_fig = px.scatter_mapbox(
                    clustered_df,
                    lat='pickup_latitude',
                    lon='pickup_longitude',
                    color='cluster',
                    size='auction_winning_price',
                    mapbox_style="stamen-terrain",
                    zoom=10,
                    title='Pickup Location Clusters'
                )
                st.plotly_chart(cluster_fig)
        
        with tab3:
            st.header("Grid Analysis")
            grid_size = st.slider("Grid size (degrees)", 0.001, 0.1, 0.01, 0.001)
            
            # Create grid
            grid_df = create_grid(merged_df.copy(), grid_size)
            
            # Aggregate by grid
            pickup_grid_prices = grid_df.groupby(
                ['pickup_grid_x', 'pickup_grid_y']
            )['auction_winning_price'].agg(['mean', 'count']).reset_index()
            
            # Create grid visualization
            grid_fig = px.scatter(
                pickup_grid_prices,
                x='pickup_grid_x',
                y='pickup_grid_y',
                size='count',
                color='mean',
                title='Average Auction Prices by Grid',
                labels={
                    'pickup_grid_x': 'Grid X',
                    'pickup_grid_y': 'Grid Y',
                    'mean': 'Average Price (€)',
                    'count': 'Number of Bookings'
                }
            )
            st.plotly_chart(grid_fig)
            
            # Show highest price areas
            st.subheader("Top 5 Highest Price Areas")
            top_areas = pickup_grid_prices.nlargest(5, 'mean')
            st.dataframe(top_areas)
    
    except Exception as e:
        st.error(f"Error in analysis: {str(e)}")
        st.write("Please ensure the data contains the required geographical information.")

if __name__ == "__main__":
    main()
