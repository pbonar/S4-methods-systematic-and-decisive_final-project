import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# Defining columns to use for input and output
IN_COLUMNS = [
    'COUTYP4', 'AGE3', 'SEXIDENT', 'IRSEX', 'NEWRACE2',
    'INCOME', 'CIGFLAG', 'ALCFLAG'
]

OUT_COLUMNS = [
    'MJEVER', 'COCEVER', 'HEREVER', 'LSD', 'PCP', 'PEYOTE',
    'MESC', 'PSILCY', 'ECSTMOLLY', 'KETMINESK', 'DMTAMTFXY',
    'SALVIADIV', 'HALLUCEVR', 'INHALEVER', 'METHAMEVR',
    'PNRNMLIF', 'TRQNMLIF', 'STMANYLIF', 'SEDANYLIF',
    'HLTINALC', 'HLTINDRG'
]

# Descriptions of possible options for each input column
INPUT_DESCRIPTIONS = {
    'COUTYP4': {
        'description': 'Miejsce zamieszkania',
        'options': {
            1: 'Duża Metropolia',
            2: 'Mała Metropolia',
            3: 'Obszar Pozamiejski'
        }
    },
    'AGE3': {
        'description': 'Grupa wiekowa',
        'options': {
            4: '18-20 lat',
            5: '21-23 lat',
            6: '24-25 lat',
            7: '26-29 lat',
            8: '30-34 lat',
            9: '35-49 lat',
            10: '50-64 lat',
            11: '65+ lat'
        }
    },
    'SEXIDENT': {
        'description': 'Preferencje seksualne',
        'options': {
            1: 'Heteroseksualni',
            2: 'Homoseksualni',
            3: 'Biseksualni'
        }
    },
    'IRSEX': {
        'description': 'Płeć',
        'options': {
            1: 'Mężczyźni',
            2: 'Kobiety'
        }
    },
    'NEWRACE2': {
        'description': 'Pochodzenie etniczne',
        'options': {
            1: 'Białoskórzy',
            2: 'Czarnoskórzy',
            3: 'Indianie',
            4: 'Hawajczycy',
            5: 'Azjaci',
            6: 'Więcej niż jedna rasa',
            7: 'Latynosi'
        }
    },
    'INCOME': {
        'description': 'Całkowity zarobek rodzinny rocznie',
        'options': {
            1: 'Mniej niż 20,000 $',
            2: '20,000-49,999 $',
            3: '50,000-74,999 $',
            4: 'Więcej niż 75,000$'
        }
    },
    'CIGFLAG': {
        'description': 'Papierosy',
        'options': {
            1: 'Brano',
            2: 'Nie brano'
        }
    },
    'ALCFLAG': {
        'description': 'Alkohol',
        'options': {
            1: 'Brano',
            2: 'Nie brano'
        }
    }
}

def train_random_forest(csv_filepath: str):
    # Load data
    df = pd.read_csv(csv_filepath)

    # Separate features and target variables
    X = df[IN_COLUMNS]
    y = df[OUT_COLUMNS]

    # Scale input features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Add column names to scaler
    scaler_columns = IN_COLUMNS
    scaler.mean_ = pd.Series(scaler.mean_, index=scaler_columns)
    scaler.scale_ = pd.Series(scaler.scale_, index=scaler_columns)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Initialize and train the Random Forest model for each target column
    rf_models = {}
    for column in OUT_COLUMNS:
        rf_model = RandomForestClassifier(n_estimators=10, random_state=42)
        rf_model.fit(X_train, y_train[column])
        rf_models[column] = rf_model

    # Return the trained models and the scaler
    return rf_models, scaler

def get_user_input():
    user_input = []
    for feature, description in INPUT_DESCRIPTIONS.items():
        print(f"{feature} ({description['description']}):")
        for key, value in description['options'].items():
            print(f"  {key}: {value}")
        while True:
            try:
                value = int(input(f"Wprowadz wartosc dla {feature}: "))
                if value in description['options']:
                    user_input.append(value)
                    break
                else:
                    print("Niepoprawna wartosc. Sprobuj ponownie.")
            except ValueError:
                print("Niepoprawna watosc. Wprowadz numer.")
    return user_input

def predict_with_random_forest(rf_models, scaler):
    # Get user input
    user_input = get_user_input()

    # Create DataFrame for user input
    user_input_df = pd.DataFrame([user_input], columns=IN_COLUMNS)

    # Scale the user input
    user_input_scaled = scaler.transform(user_input_df)

    # Make predictions using the trained models
    predictions = {}
    for column, model in rf_models.items():
        prediction = model.predict(user_input_scaled)
        prediction_label = "TAK" if prediction[0] == 1 else "NIE"
        predictions[column] = prediction_label

    return predictions

if __name__ == "__main__":
    # Path to the validated data CSV file
    csv_filepath = "../data/NSDUH_2022_selected_columns_validated.csv"

    # Train the Random Forest models
    rf_models, scaler = train_random_forest(csv_filepath)

    # Get predictions for user input
    predictions = predict_with_random_forest(rf_models, scaler)

    # Print out the predictions
    print("\nPredykcje dla wprowadzonych danych:")
    for column, prediction in predictions.items():
        print(f"{column}: {prediction}")
