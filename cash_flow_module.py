import os
import pandas as pd
from sklearn.linear_model import LinearRegression
from sqlalchemy import create_engine
from sklearn.metrics import mean_absolute_error
import joblib

# Function to extract daily transaction data for a specific user from the database
def extract_daily_data(db_url, user_id):
    try:
        engine = create_engine(db_url)
        query = f"SELECT date, amount FROM daily_transactions WHERE user_id = {user_id}"
        daily_data = pd.read_sql(query, engine)
        
        # Ensure 'date' is in datetime format
        daily_data['date'] = pd.to_datetime(daily_data['date'])
        return daily_data
    except Exception as e:
        print(f"Error fetching data: {e}")
        return None

# Function to aggregate daily data into monthly data
def aggregate_to_monthly(daily_data):
    daily_data['month'] = daily_data['date'].dt.to_period('M')
    monthly_data = daily_data.groupby('month')['amount'].sum().reset_index()
    monthly_data.columns = ['month', 'total_cash_flow']
    monthly_data['month'] = monthly_data['month'].dt.to_timestamp()  # Convert to timestamp for datetime compatibility
    return monthly_data

# Function to prepare features and target for training
def prepare_data(monthly_data):
    monthly_data['month_num'] = monthly_data['month'].dt.month
    monthly_data['year'] = monthly_data['month'].dt.year
    
    X = monthly_data[['month_num', 'year']]
    y = monthly_data['total_cash_flow']
    
    return X, y

# Function to train the model, save it, and make predictions
def train_and_predict(X, y, model_path="cash_flow_model.joblib"):
    # Initialize the Linear Regression model
    model = LinearRegression()
    model.fit(X, y)  # Train the model on the entire dataset

    # Save the model
    joblib.dump(model, model_path)
    
    # Make predictions on the same data
    predictions = model.predict(X)
    
    return predictions

# Main function to execute the workflow
def main(user_id, db_url=None):
    # Fetch DB URL from environment variable if not provided
    db_url = db_url or os.getenv('DB_URL')
    
    if not db_url:
        print("Database URL is missing. Set it as an environment variable or pass it as a parameter.")
        return
    
    # Step 1: Data extraction
    daily_data = extract_daily_data(db_url, user_id)
    if daily_data is None or daily_data.empty:
        print("No data fetched. Exiting...")
        return
    
    # Step 2: Aggregate to monthly data
    monthly_data = aggregate_to_monthly(daily_data)
    
    # Step 3: Prepare features and target
    X, y = prepare_data(monthly_data)
    
    # Step 4: Train the model and get predictions
    predictions = train_and_predict(X, y)
    
    # Output predictions
    print("Predictions:", predictions)

if __name__ == "__main__":
    # Set the user_id and optionally pass db_url if it's not set in the environment
    user_id = 1  # Replace with the actual user ID
    main(user_id)
