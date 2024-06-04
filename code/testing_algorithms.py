import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score

# Define columns to use for input and output
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

# Rounding the values to 0.___
def round_dict_values(d, decimal_places=3):
    for key, value in d.items():
        if isinstance(value, dict):
            round_dict_values(value, decimal_places)
        elif isinstance(value, (float, int)):
            d[key] = round(value, decimal_places)

# Training and evaluating different models
def train_and_evaluate(csv_filepath: str):
    # Loading data
    df = pd.read_csv(csv_filepath)

    # Separating features and target variables
    X = df[IN_COLUMNS]
    y = df[OUT_COLUMNS]

    # Splitting data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scaling input features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Initializing the models
    models = {
        'Logistic Regression': LogisticRegression(max_iter=200),
        'Random Forest': RandomForestClassifier(n_estimators=100),
        'SVM': SVC(kernel='linear', max_iter=4000)
    }

    # Training and evaluating each model for each target column
    results = {}
    average_accuracies = {}
    for model_name, model in models.items():
        results[model_name] = {}
        accuracies = []
        for column in OUT_COLUMNS:
            model.fit(X_train_scaled, y_train[column])
            y_pred = model.predict(X_test_scaled)

            accuracy = round(accuracy_score(y_test[column], y_pred), 3)
            accuracies.append(accuracy)
            report = classification_report(y_test[column], y_pred, output_dict=True, zero_division=0)
            round_dict_values(report, 3)
            results[model_name][column] = {
                'accuracy': accuracy,
                'report': report
            }

        # Calculating the average accuracy for the current model
        average_accuracy = round(sum(accuracies) / len(accuracies), 3)
        average_accuracies[model_name] = average_accuracy

    return results, average_accuracies

# Creating a summary table for our models
def create_summary_table(results, average_accuracies):
    tables = {}
    for model_name, model_results in results.items():
        rows = []
        for column, metrics in model_results.items():
            accuracy = metrics['accuracy']
            report = metrics['report']
            # Fetch dynamically all class keys from the report
            available_classes = report.keys() - {'accuracy', 'macro avg', 'weighted avg'}
            row = {
                'Metric': column,
                'Accuracy': accuracy,
            }
            for cls in available_classes:
                row[f'Precision ({cls})'] = report[cls]['precision']
                row[f'Recall ({cls})'] = report[cls]['recall']
                row[f'F1-Score ({cls})'] = report[cls]['f1-score']
            rows.append(row)

        df_table = pd.DataFrame(rows)
        avg_accuracy = average_accuracies[model_name]
        avg_row = pd.DataFrame([{
            'Metric': 'Average',
            'Accuracy': avg_accuracy,
            **{f'Precision ({cls})': '-' for cls in available_classes},
            **{f'Recall ({cls})': '-' for cls in available_classes},
            **{f'F1-Score ({cls})': '-' for cls in available_classes}
        }])
        df_table = pd.concat([df_table, avg_row], ignore_index=True)
        tables[model_name] = df_table
    return tables


if __name__ == "__main__":
    # Path to the validated data CSV file
    csv_filepath = "../data/NSDUH_2022_selected_columns_validated.csv"

    # Train the models and evaluate them
    results, average_accuracies = train_and_evaluate(csv_filepath)

    # Create summary tables for each model
    tables = create_summary_table(results, average_accuracies)

    # Display the results
    for model_name, df_table in tables.items():
        print(f"\nModel: {model_name}")
        print(df_table.to_string(index=False))
