import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

def load_data(file_path):
    df = pd.read_csv(file_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df.columns = [col.strip() for col in df.columns]
    print("Here's what the data looks like:")
    print(df.head())
    
    numeric_cols = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
    
    for col in numeric_cols:
        try:
            temp = df[col].astype(str).str.replace(',', '')
            non_numeric = ~temp.str.match(r'^[-+]?[0-9]*\.?[0-9]+$')
            if non_numeric.any():
                bad_values = df.loc[non_numeric, col].unique()
                print(f"Warning: Found non-numeric values in {col}: {bad_values}")
                print(f"These will be converted to NaN")
            
            df[col] = pd.to_numeric(temp, errors='coerce')
            
        except Exception as e:
            print(f"Error processing column {col}: {str(e)}")
            print(f"Setting {col} to NaN where values are invalid")
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', ''), errors='coerce')
    
    orig_len = len(df)
    df = df.dropna(subset=['Open', 'Close'])
    dropped = orig_len - len(df)
    if dropped > 0:
        print(f"Dropped {dropped} rows with missing values in key columns")
    
    print("Data loaded successfully!")
    return df

def add_indicators(data):
    df = data.copy()
    df['MA_5'] = df['Close'].rolling(window=5).mean()
    df['MA_10'] = df['Close'].rolling(window=10).mean()
    df['MA_20'] = df['Close'].rolling(window=20).mean()
    df['MA_50'] = df['Close'].rolling(window=50).mean()
    df['Daily_Return'] = df['Close'].pct_change()
    delta = df['Close'].diff()
    gains = delta.copy()
    losses = delta.copy()
    gains[gains < 0] = 0
    losses[losses > 0] = 0
    losses = abs(losses)
    avg_gain = gains.rolling(window=14).mean()
    avg_loss = losses.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))
    df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['Next_Day_Close'] = df['Close'].shift(-1)
    df = df.dropna()
    print(f"Added indicators. New data shape: {df.shape}")
    return df

def prepare_data(data):
    features = ['MA_5', 'MA_10', 'MA_20', 'MA_50', 'Daily_Return', 'RSI', 'MACD', 'MACD_Signal']
    X = data[features]
    y = data['Next_Day_Close']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print("Data split into training and testing sets")
    print(f"Training set size: {X_train.shape}")
    print(f"Testing set size: {X_test.shape}")
    return X_train, X_test, y_train, y_test

def build_model(X_train, X_test, y_train, y_test):
    model = LinearRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, predictions)
    print("Model Results:")
    print(f"Mean Squared Error: {mse}")
    print(f"Root Mean Squared Error: {rmse}")
    print(f"R² Score: {r2}")
    importance = pd.DataFrame({
        'Feature': X_train.columns,
        'Importance': model.coef_
    })
    importance = importance.sort_values('Importance', ascending=False)
    print("\nFeature Importance:")
    print(importance)
    return model, predictions

def plot_results(y_test, predictions):
    plt.figure(figsize=(10, 6))
    plt.plot(y_test.values, label='Actual Price', color='blue')
    plt.plot(predictions, label='Predicted Price', color='red')
    plt.title('Stock Price Prediction')
    plt.xlabel('Data Points')
    plt.ylabel('Price')
    plt.legend()
    plt.show()
    errors = y_test.values - predictions
    plt.figure(figsize=(10, 6))
    plt.hist(errors, bins=25)
    plt.title('Prediction Errors')
    plt.xlabel('Error')
    plt.ylabel('Count')
    plt.show()

def main():
    file_path = 'Netflix Inc. (NFLX) Stock Price 2002-2025.csv'
    data = load_data(file_path)
    data_with_indicators = add_indicators(data)
    X_train, X_test, y_train, y_test = prepare_data(data_with_indicators)
    model, predictions = build_model(X_train, X_test, y_train, y_test)
    plot_results(y_test, predictions)
    print("Analysis finished!")

if __name__ == "__main__":
    main()