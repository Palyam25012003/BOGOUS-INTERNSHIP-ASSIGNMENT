import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
import streamlit as st
from streamlit_option_menu import option_menu
from sklearn.impute import SimpleImputer
from feature_engine.outliers import OutlierTrimmer

# Initialize session state variables
session_variables = ['execute_code', 'data_loaded', 'preprocessed']
for var in session_variables:
    if var not in st.session_state:
        st.session_state[var] = None

class HotelBookingAnalysis:
    def __init__(self):
        self.data = None
        self.load_data()
        
    def load_data(self):
        if st.session_state.data_loaded is None:
            try:
                self.data = pd.read_csv("hotel_bookings.csv")
                st.session_state.data_loaded = True
            except FileNotFoundError:
                st.error("File 'hotel_bookings.csv' not found. Please upload the file.")
                uploaded_file = st.file_uploader("Upload hotel_bookings.csv", type="csv")
                if uploaded_file is not None:
                    self.data = pd.read_csv(uploaded_file)
                    st.session_state.data_loaded = True
    
    def show_basic_info(self):
        st.subheader("Basic Dataset Information")
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("**First 5 rows:**")
            st.write(self.data.head())
        with col2:
            st.write("**Last 5 rows:**")
            st.write(self.data.tail())
        
        st.write("**Random Sample (20% of data):**")
        sample_data = self.data.sample(frac=0.2)
        st.write(sample_data)
        st.write(f"Sample shape: {sample_data.shape}")
        
        st.write("**Columns and Data Types:**")
        st.write(self.data.dtypes)
        
        st.write("**Numerical Columns:**")
        num_data = self.data.select_dtypes(include=['int64', 'float64'])
        st.write(num_data.columns.tolist())
        st.write(f"Numerical data shape: {num_data.shape}")
        
        st.write("**Categorical Columns:**")
        cat_data = self.data.select_dtypes(include=['object'])
        st.write(cat_data.columns.tolist())
        st.write(f"Categorical data shape: {cat_data.shape}")
    
    def show_missing_data(self):
        st.subheader("Missing Data Analysis")
        
        st.write("**Total Missing Values:**")
        st.write(self.data.isna().sum().sum())
        
        st.write("**Missing Values by Column:**")
        missing_data = self.data.isna().sum().reset_index()
        missing_data.columns = ['Column', 'Missing Values']
        st.write(missing_data[missing_data['Missing Values'] > 0])
        
        st.write("**Missing Data Visualization:**")
        fig, ax = plt.subplots(figsize=(10, 6))
        msno.matrix(self.data, ax=ax)
        st.pyplot(fig)
    
    def preprocess_data(self):
        st.subheader("Data Preprocessing")
        
        if st.button("Preprocess Data"):
            # Handle missing values
            st.write("**Handling Missing Values...**")
            self.data[['children', 'agent', 'company']] = SimpleImputer(strategy='median').fit_transform(
                self.data[['children', 'agent', 'company']])
            self.data['country'] = self.data['country'].fillna("Unknown")
            
            # Remove duplicates
            st.write("**Removing Duplicates...**")
            self.data = self.data.drop_duplicates()
            
            # Handle outliers
            st.write("**Handling Outliers...**")
            numeric_cols = self.data.select_dtypes(include=['int64', 'float64']).columns
            for col in numeric_cols:
                q1 = self.data[col].quantile(0.25)
                q3 = self.data[col].quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                self.data[col] = np.where(
                    (self.data[col] < lower_bound) | (self.data[col] > upper_bound),
                    self.data[col].mean(),
                    self.data[col]
                )
            
            st.session_state.preprocessed = True
            st.success("Data preprocessing completed!")
            
            # Show post-processing info
            self.show_basic_info()
            self.show_missing_data()
    
    def perform_eda(self):
        st.subheader("Exploratory Data Analysis")
        
        if not st.session_state.preprocessed:
            st.warning("Please preprocess the data first for better results.")
            return
        
        analysis_option = st.selectbox(
            "Select Analysis",
            ["Revenue Trends", "Cancellation Rate", "Geographical Distribution", 
             "Lead Time Distribution", "Room Type Analysis"]
        )
        
        if analysis_option == "Revenue Trends":
            self.revenue_trends()
        elif analysis_option == "Cancellation Rate":
            self.cancellation_rate()
        elif analysis_option == "Geographical Distribution":
            self.geographical_distribution()
        elif analysis_option == "Lead Time Distribution":
            self.lead_time_distribution()
        elif analysis_option == "Room Type Analysis":
            self.room_type_analysis()
    
    def revenue_trends(self):
        st.write("## Revenue Trends Over Time")
        
        # Create arrival date
        self.data['arrival_date'] = pd.to_datetime(
            self.data['arrival_date_year'].astype(str) + '-' +
            self.data['arrival_date_month'] + '-' +
            self.data['arrival_date_day_of_month'].astype(str),
            format='%Y-%B-%d',
            errors='coerce'
        )
        
        # Calculate revenue
        self.data['revenue'] = self.data['adr'] * (
            self.data['stays_in_week_nights'] + self.data['stays_in_weekend_nights'])
        
        # Group by month-year
        self.data['month_year'] = self.data['arrival_date'].dt.to_period('M')
        revenue_trend = self.data.groupby('month_year')['revenue'].sum().reset_index()
        revenue_trend['month_year'] = revenue_trend['month_year'].astype(str)
        
        # Display results
        st.write("### Monthly Revenue Summary")
        st.dataframe(revenue_trend)
        
        st.write("### Descriptive Statistics")
        st.write(revenue_trend['revenue'].describe())
        
        # Plot
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.lineplot(data=revenue_trend, x='month_year', y='revenue', ax=ax)
        ax.set_title('Revenue Trends Over Time')
        ax.set_xlabel('Month-Year')
        ax.set_ylabel('Total Revenue')
        plt.xticks(rotation=45)
        st.pyplot(fig)
    
    def cancellation_rate(self):
        st.write("## Cancellation Rate Analysis")
        
        total_bookings = len(self.data)
        canceled_bookings = self.data['is_canceled'].sum()
        cancellation_rate = (canceled_bookings / total_bookings) * 100
        
        # Display results
        st.write(f"**Total Bookings:** {total_bookings:,}")
        st.write(f"**Canceled Bookings:** {canceled_bookings:,}")
        st.write(f"**Cancellation Rate:** {cancellation_rate:.2f}%")
        
        if 'hotel' in self.data.columns:
            st.write("### Cancellation Rate by Hotel Type")
            cancel_by_hotel = self.data.groupby('hotel')['is_canceled'].mean() * 100
            st.write(cancel_by_hotel)
        
        # Plot
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.pie([canceled_bookings, total_bookings - canceled_bookings],
                labels=['Canceled', 'Not Canceled'],
                autopct='%1.1f%%',
                startangle=90)
        ax.set_title(f'Cancellation Rate ({cancellation_rate:.2f}%)')
        st.pyplot(fig)
    
    def geographical_distribution(self):
        st.write("## Geographical Distribution Analysis")
        
        # Top 10 countries
        country_dist = self.data['country'].value_counts().head(10)
        country_percentage = (country_dist / len(self.data)) * 100
        
        # Display results
        st.write("### Top 10 Countries by Bookings")
        st.write(country_dist)
        
        st.write("### Percentage of Total Bookings")
        st.write(country_percentage.round(2))
        
        # Plot
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.barplot(x=country_dist.index, y=country_dist.values, ax=ax)
        ax.set_title('Top 10 Countries by Number of Bookings')
        ax.set_xlabel('Country Code')
        ax.set_ylabel('Number of Bookings')
        st.pyplot(fig)
    
    def lead_time_distribution(self):
        st.write("## Lead Time Distribution Analysis")
        
        # Calculate statistics
        stats = {
            'Mean': self.data['lead_time'].mean(),
            'Median': self.data['lead_time'].median(),
            'Std Dev': self.data['lead_time'].std(),
            'Min': self.data['lead_time'].min(),
            'Max': self.data['lead_time'].max()
        }
        
        # Display results
        st.write("### Lead Time Statistics (in days)")
        st.write(pd.DataFrame.from_dict(stats, orient='index', columns=['Value']))
        
        st.write("### Percentile Information")
        st.write(self.data['lead_time'].describe(percentiles=[.25, .5, .75, .9, .95]))
        
        # Plot
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.histplot(self.data['lead_time'], bins=50, kde=True, ax=ax)
        ax.set_title('Distribution of Booking Lead Time (Days)')
        ax.set_xlabel('Lead Time (Days)')
        ax.set_ylabel('Number of Bookings')
        st.pyplot(fig)
    
    def room_type_analysis(self):
        st.write("## Room Type Analysis")
        
        # Compare reserved vs assigned room types
        room_comparison = pd.crosstab(
            self.data['reserved_room_type'], 
            self.data['assigned_room_type']
        )
        
        # Calculate mismatch rate
        mismatch_rate = (self.data['reserved_room_type'] != self.data['assigned_room_type']).mean() * 100
        
        # Display results
        st.write("### Room Type Assignment Matrix")
        st.write(room_comparison)
        
        st.write(f"**Mismatch Rate:** {mismatch_rate:.2f}%")
        
        # Plot
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(room_comparison, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_title('Reserved vs Assigned Room Types')
        ax.set_xlabel('Assigned Room Type')
        ax.set_ylabel('Reserved Room Type')
        st.pyplot(fig)

def main():
    st.set_page_config(page_title="Hotel Booking Analysis", layout="wide")
    st.title("Hotel Booking Data Analysis")
    
    # Initialize analysis class
    if 'analysis' not in st.session_state:
        st.session_state.analysis = HotelBookingAnalysis()
    
    analysis = st.session_state.analysis
    
    # Sidebar navigation
    with st.sidebar:
        selected = option_menu(
            "Main Menu",
            ["Dataset Overview", "Data Preprocessing", "Exploratory Analysis"],
            icons=["clipboard-data", "gear", "graph-up"],
            default_index=0,
        )
    
    # Display selected page
    if selected == "Dataset Overview":
        st.header("Dataset Overview")
        analysis.show_basic_info()
        analysis.show_missing_data()
    
    elif selected == "Data Preprocessing":
        st.header("Data Preprocessing")
        analysis.preprocess_data()
    
    elif selected == "Exploratory Analysis":
        st.header("Exploratory Data Analysis")
        analysis.perform_eda()

if __name__ == "__main__":
    main()
