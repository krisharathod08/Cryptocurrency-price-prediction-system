from flask import Flask, render_template, request, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField, BooleanField,SelectField
from wtforms.validators import DataRequired, Length
import pandas as pd
from datetime import datetime, timedelta
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
import numpy as np
import requests
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__, static_url_path='/static', static_folder='static')
app.config['SECRET_KEY'] = 'your_secret_key'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///site.db'
db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(20), unique=True, nullable=False)
    password = db.Column(db.String(60), nullable=False)
    is_admin = db.Column(db.Boolean, default=False)

class RegistrationForm(FlaskForm):
    username = StringField('username', validators=[DataRequired(), Length(min=2, max=20)])
    password = PasswordField('password', validators=[DataRequired()])
    submit = SubmitField('Sign Up')

# Fetch user by username
def fetch_user_by_username(username):
    return User.query.filter_by(username=username).first()

# Fetch user by ID
def fetch_user_by_id(user_id):
    return User.query.get(int(user_id))

# Create a new user
def create_user(username, password, is_admin=False):
    new_user = User(username=username, password=password, is_admin=is_admin)
    db.session.add(new_user)
    db.session.commit()

# Fetch all users
def fetch_all_users():
    return User.query.all()

@app.route('/')
def index():
    if current_user.is_authenticated:
        return redirect(url_for('admin_dashboard')) if current_user.is_admin else redirect(url_for('price_prediction'))
    else:
        return redirect(url_for('login'))

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        user = fetch_user_by_username(username)

        if user and user.password == password:
            login_user(user)
            if user.is_admin:
                return redirect(url_for('admin_dashboard'))
            else:
                return redirect(url_for('price_prediction'))
        else:
            flash('Invalid username or password. Please try again.', 'error')

    return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    form = RegistrationForm()

    if form.validate_on_submit():
        username = form.username.data
        password = form.password.data

        existing_user = fetch_user_by_username(username)
        if existing_user:
            flash('Username already exists. Please choose a different username.', 'error')
        else:
            create_user(username, password)
            flash('Account created successfully. Please log in.', 'success')
            return redirect(url_for('login'))

    return render_template('signup.html', form=form)

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

class ModelSelectionForm(FlaskForm):
    model = SelectField('Choose a Model', choices=[('linear_regression', 'Linear Regression'), ('decision_tree', 'Decision Tree')])
    cryptocurrency = SelectField('Choose a Cryptocurrency', choices=[('bitcoin', 'Bitcoin'), ('matic-network', 'Matic')])


def fetch_cryptocurrency_data(symbol, days=365):
    # Placeholder: Use the CoinGecko API or any other data source to fetch cryptocurrency data
    # Replace the following code with your actual data fetching logic
    end_date = datetime.now().strftime("%d-%m-%Y")
    start_date = (datetime.now() - timedelta(days=days)).strftime("%d-%m-%Y")

    # Your code to fetch cryptocurrency data here

    # Placeholder: Sample data (replace with actual data fetching logic)
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    prices = np.random.rand(len(dates)) * 1000
    df = pd.DataFrame({'Date': dates, f'Close_Price_{symbol.upper()}': prices})
    return df

def fetch_cryptocurrency_data(symbol, days=365):
    end_date = datetime.now().strftime("%d-%m-%Y")
    start_date = (datetime.now() - timedelta(days=days)).strftime("%d-%m-%Y")

    url = f'https://api.coingecko.com/api/v3/coins/{symbol}/market_chart'
    params = {
        'vs_currency': 'usd',
        'days': days,
        'interval': 'daily',
    }

    response = requests.get(url, params=params)
    data = response.json()

    if 'prices' not in data:
        raise ValueError(f"Unexpected response from CoinGecko API for {symbol}: {data}")

    df = pd.DataFrame(data['prices'], columns=['timestamp', 'close_price'])

    if df.empty:
        raise ValueError(f"No price data available for {symbol} in the last {days} days")

    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df = df.rename(columns={'timestamp': 'Date', 'close_price': f'Close_Price_{symbol.upper()}'})
    
    return df

def preprocess_data(data, symbol):
    data['Date'] = pd.to_datetime(data['Date']).dt.date  # Extract only the date
    data.set_index('Date', inplace=True)
    data.sort_index(inplace=True)
    data = data.rename(columns={'close_price': f'Close_Price_{symbol.upper()}'})
    return data


def train_linear_regression(X_train, y_train, X_test):
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)

    # Ensure that X_test has the same length as y_train
    X_test = X_test.iloc[:len(y_train)]

    return lr_model, X_test

def train_decision_tree(X_train, y_train, X_test):
    dt_model = DecisionTreeRegressor()
    dt_model.fit(X_train, y_train)

    # Ensure that X_test has the same length as y_train
    X_test = X_test.iloc[:len(y_train)]

    return dt_model, X_test

def predict(model, X):
    return model.predict(X)



def plot_predictions(data, predictions, title, crypto_symbol):
    plt.figure(figsize=(12, 6))
    plt.plot(data.index[-len(data):], data[f'Close_Price_{crypto_symbol.upper()}'], label='Actual Prices', color='blue')
    plt.plot(data.index[-len(predictions):], predictions, label=f'{title} Predictions', color='red')
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Close Price')
    plt.legend()

    img_bytes = io.BytesIO()
    plt.savefig(img_bytes, format='png')
    plt.cla()
    plt.clf()
    plt.close()
    

    return img_bytes


@app.route('/price_prediction', methods=['GET', 'POST'])
def price_prediction():
    form = ModelSelectionForm()

    if form.validate_on_submit():
        selected_cryptocurrency = form.cryptocurrency.data
        selected_model = form.model.data

        # Fetch historical price data for the selected cryptocurrency
        crypto_data = fetch_cryptocurrency_data(selected_cryptocurrency)
        crypto_data = preprocess_data(crypto_data, selected_cryptocurrency)

        # Create features and target for the cryptocurrency data
        crypto_data['Target'] = crypto_data[f'Close_Price_{selected_cryptocurrency.upper()}'].shift(-1)
        features = crypto_data[[f'Close_Price_{selected_cryptocurrency.upper()}']]
        target = crypto_data['Target'].dropna()

        # Manually split the dataset into training and testing sets
        split_index = int(len(features) * 0.8)
        X_train, y_train = features.iloc[:split_index], target.iloc[:split_index]
        X_test, y_test = features.iloc[split_index:], target.iloc[split_index:]

        # Train the selected model
        if selected_model == 'linear_regression':
            model, X_test_model = train_linear_regression(X_train, y_train, X_test)
        elif selected_model == 'decision_tree':
            model, X_test_model = train_decision_tree(X_train, y_train, X_test)
        else:
            return "Invalid model selection"

        # Make predictions
        predictions = predict(model, X_test_model)

        # Get the base64-encoded image string
        img_bytes = plot_predictions(crypto_data, predictions, selected_model, selected_cryptocurrency)

        img_str = base64.b64encode(img_bytes.getvalue()).decode('utf-8')

        return render_template('price_prediction.html', form=form, img_str=img_str)
    return render_template('price_prediction.html', form=form)

@app.route('/admin_dashboard')
@login_required
def admin_dashboard():
    if current_user.is_authenticated and current_user.is_admin:
        return render_template('admin_dashboard.html', users=fetch_all_users())
    else:
        flash('You are not authorized to access the admin dashboard.', 'error')
        return redirect(url_for('index'))

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True, port=5001,threaded=False)
