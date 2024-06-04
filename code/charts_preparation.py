import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List

# Function for creating a histogram of age groups
def histograph_age(df: pd.DataFrame) -> None:
    # Defining colors and labels for age groups
    colors = ['skyblue', 'lightcoral', 'lightgreen', 'gold', 'lightsalmon', 'lightseagreen', 'lightpink',
              'lightsteelblue', 'lightyellow', 'lightgrey', 'lightcyan']
    labels = ['4 - 18-20 lat',
              '5 - 21-23 lat',
              '6 - 24-25 lat',
              '7 - 26-29 lat',
              '8 - 30-34 lat',
              '9 - 35-49 lat',
              '10- 50-64 lat',
              '11- 65+ lat']

    # Getting counts and sorting them
    age_counts = df['AGE3'].value_counts().sort_index()

    # Creating bars for the histogram
    bars = plt.bar(age_counts.index, age_counts.values, color=colors[:len(age_counts)])

    # Setting ticks, labels, and title
    plt.xticks(age_counts.index, range(4, 12), rotation=45, ha='right')
    plt.ylabel('Liczba Uczestników')
    plt.xlabel('Grupa wiekowa')
    plt.title('Rozkład uczestników według grupy wiekowej (AGE3)')

    # Creating legend
    plt.legend(bars, labels, loc='upper left')

    # Saving and displaying the plot
    plt.savefig('../charts/histographs_basic/histogram_age3.jpg', dpi=300)
    plt.show()


# Function for creating a histogram of place types
def histograph_coutyp4(df: pd.DataFrame) -> None:
    # Dividing the rows into groups
    large_metro = df[df['COUTYP4'] == 1]
    small_metro = df[df['COUTYP4'] == 2]
    non_metro = df[df['COUTYP4'] == 3]

    # Creating histogram
    plt.hist([large_metro['COUTYP4'], small_metro['COUTYP4'], non_metro['COUTYP4']], bins=range(1, 5), rwidth=0)

    # Adding labels, titles and ticks
    plt.xticks([1, 2, 3], ['Duża Metropolia', 'Mała Metropolia', 'Obszar Pozamiejski'])
    plt.title('Rozkład ilości uczestników według miejsca zamieszkania (COUTYP4)')
    plt.ylabel('Liczba uczestników')

    # Adjusting the bars
    counts = [large_metro.shape[0], small_metro.shape[0], non_metro.shape[0]]
    bars = plt.bar(range(1, 4), counts, color=['skyblue', 'lightgreen', 'lightsalmon'], width=0.5, align='center')
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval + 0.1, int(yval), ha='center', va='bottom')

    # Saving and displaying the plot
    plt.savefig('../charts/histographs_basic/histogram_coutyp4.jpg', dpi=300)
    plt.show()


# Function for creating a histogram of sexual identity
def histograph_sexident(df: pd.DataFrame) -> None:
    # Dividing the rows into groups
    homo = df[df['SEXIDENT'] == 1]
    hetero = df[df['SEXIDENT'] == 2]
    bi = df[df['SEXIDENT'] == 3]

    # Creating the histogram
    plt.hist([homo['SEXIDENT'], hetero['SEXIDENT'], bi['SEXIDENT']], bins=range(1, 5), rwidth=0)

    # Adding ticks, titles and labels
    plt.xticks([1, 2, 3], ['Heteroseksualni', 'Homoseksualni', 'Biseksualni'])
    plt.title('Rozkład ilości uczestników według preferencji seksualnych (SEXIDENT)')
    plt.ylabel('Liczba uczestników')

    # Adjusting the bars
    counts = [homo.shape[0], hetero.shape[0], bi.shape[0]]
    bars = plt.bar(range(1, 4), counts, color=['skyblue', 'lightgreen', 'lightsalmon'], width=0.5, align='center')
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval + 0.1, int(yval), ha='center', va='bottom')

    # Saving and displaying the plot
    plt.savefig('../charts/histographs_basic/histogram_sexident.jpg', dpi=300)
    plt.show()


# Function for creating a histogram of gender
def histograph_irsex(df: pd.DataFrame) -> None:
    # Dividing the rows into groups
    male = df[df['IRSEX'] == 1]
    female = df[df['IRSEX'] == 2]

    # Creating the histogram
    plt.hist([male['IRSEX'], female['IRSEX']], bins=range(1, 4), color=['skyblue', 'pink'], rwidth=0.3, align='left')

    # Adding ticks, titles and labels
    plt.xticks([1, 2], ['Mężczyźni', 'Kobiety'])
    plt.title('Rozkład ilości uczestników według płci (IRSEX)')
    plt.ylabel('Liczba uczestników')

    # Adjusting the bars
    bars = plt.bar(range(1, 3), [male.shape[0], female.shape[0]], color=['skyblue', 'pink'], width=0.3)
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval + 0.1, round(yval, 2), ha='center', va='bottom')

    # Saving and displaying the plot
    plt.savefig('../charts/histographs_basic/histogram_irsex.jpg', dpi=300)
    plt.show()


# Function for creating a histogram of ethnic background
def histograph_newrace2(df: pd.DataFrame) -> None:
    # Defining colors and labels for ethnic groups
    colors = ['skyblue', 'lightcoral', 'lightgreen', 'gold', 'lightsalmon', 'lightseagreen', 'lightpink']
    labels = [
        '1 - Białoskórzy',
        '2 - Czarnoskórzy',
        '3 - Indianie',
        '4 - Hawajczycy',
        '5 - Azjaci',
        '6 - Więcej niż jedna rasa',
        '7 - Latynosi'
    ]

    # Getting ethnic background and sorting them
    race_counts = df['NEWRACE2'].value_counts().sort_index()
    bars = plt.bar(race_counts.index, race_counts.values, color=colors[:len(race_counts)])

    # Setting ticks, labels, and title
    plt.xticks(race_counts.index, range(1, 8), rotation=45, ha='right')
    plt.ylabel('Liczba uczestników')
    plt.title('Rozkład uczestników według pochodzenia etnicznego (NEWRACE2)')
    plt.legend(bars, labels, loc='upper right')

    # Saving and displaying the plot
    plt.savefig('../charts/histographs_basic/histogram_newrace2.jpg', dpi=300)
    plt.show()


# Function for creating a histogram of income levels
def histograph_income(df: pd.DataFrame) -> None:
    # Defining colors and labels for income levels
    colors = ['skyblue', 'lightcoral', 'lightgreen', 'gold', 'lightsalmon', 'lightseagreen', 'lightpink',
              'lightsteelblue', 'lightyellow', 'lightgrey', 'lightcyan']
    labels = ['1 - Mniej niż 20,000 $',
              '2 - 20,000-49,999 $',
              '3 - 50,000-74,999 $',
              '4 - Więcej niż 75,000$']

    # Getting counts and sorting them
    income_counts = df['INCOME'].value_counts().sort_index()

    # Creating bars for the histogram
    bars = plt.bar(income_counts.index, income_counts.values, color=colors[:len(income_counts)])

    # Setting ticks, labels, and title
    plt.xticks(income_counts.index, range(1, 5), ha='right')
    plt.ylabel('Liczba Uczestników')
    plt.xlabel('Całkowity zarobek rodzinny rocznie')
    plt.title('Rozkład uczestników według grupy dochodowej (INCOME)')

    # Creating legend
    plt.legend(bars, labels, loc='upper left')

    # Saving and displaying the plot
    plt.savefig('../charts/histographs_basic/histograph_income.jpg', dpi=300)
    plt.show()


# Function for creating a histogram of drug usage
def histograph_drug(df: pd.DataFrame, drug_column_name: str, drug_name: str, drug_values: List[int], drug_labels: List[str]) -> None:
    # Dividing the rows into groups
    taken = df[df[drug_column_name] == 1]
    not_taken = df[df[drug_column_name] == 2]

    # Creating the histogram
    plt.hist([taken[drug_column_name], not_taken[drug_column_name]], bins=range(1, 4), color=['lightsalmon', 'skyblue'],
             rwidth=0.3, align='left')

    # Adding ticks, titles and labels
    plt.xticks(drug_values, drug_labels)
    plt.title(f'Rozkład ilości uczestników według brania danej substancji - {drug_name} ({drug_column_name})')
    plt.ylabel('Liczba uczestników')

    # Adjusting the bars
    bars = plt.bar(range(1, 3), [taken.shape[0], not_taken.shape[0]], color=['lightsalmon', 'skyblue'], width=0.3)
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval + 0.1, round(yval, 2), ha='center', va='bottom')

    # Saving and displaying the plot
    plt.savefig(f'../charts/histographs_drugs/histographs_drug_{drug_name}.jpg', dpi=300)
    plt.show()


# Function for creating a correlation heatmap
def correlation_heatmap(df: pd.DataFrame, columns_to_correlate: List[str]) -> None:
    # Calculating correlations
    correlation_matrix = df[columns_to_correlate].corr()

    # Creating the heatmap plot
    plt.figure(figsize=(15, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title('Correlation Heatmap')

    # Saving and displaying the plot
    filename = '../charts/correlation_heatmaps/heatmap_'
    for name in columns_to_correlate:
        filename += '_' + name
    filename += '.jpg'
    plt.savefig(filename, dpi=300)
    plt.show()


def main() -> None:
    # Loading data into data frame
    filepath_data = "../data/NSDUH_2022_selected_columns_validated.csv"
    df = pd.read_csv(filepath_data, delimiter=',', low_memory=False)

    # Generating histograms of IN variables
    histograph_age(df)
    histograph_coutyp4(df)
    histograph_sexident(df)
    histograph_irsex(df)
    histograph_newrace2(df)
    histograph_income(df)

    # Generating histograms of OUT variables
    histograph_drug(df, 'CIGFLAG', 'Papierosy', [1, 2], ['Brano', 'Nie brano'])
    histograph_drug(df, 'ALCFLAG', 'Alkohol', [1, 2], ['Brano', 'Nie brano'])
    histograph_drug(df, 'MJEVER', 'Marihuana', [1, 2], ['Brano', 'Nie brano'])
    histograph_drug(df, 'COCEVER', 'Kokaina', [1, 2], ['Brano', 'Nie brano'])
    histograph_drug(df, 'HEREVER', 'Heroina', [1, 2], ['Brano', 'Nie brano'])
    histograph_drug(df, 'LSD', 'LSD', [1, 2], ['Brano', 'Nie brano'])

    # Generating correlation heatmaps
    correlation_heatmap(df, ['AGE3', 'COUTYP4', 'SEXIDENT', 'IRSEX', 'NEWRACE2', 'ALCFLAG', 'CIGFLAG'])
    correlation_heatmap(df,
                        ['COUTYP4', 'AGE3', 'SEXIDENT', 'IRSEX', 'NEWRACE2', 'HLTINALC', 'HLTINDRG', 'INCOME', 'MJEVER',
                         'COCEVER', 'HEREVER', 'LSD', 'PCP', 'PEYOTE', 'MESC', 'PSILCY', 'ECSTMOLLY', 'KETMINESK',
                         'DMTAMTFXY', 'SALVIADIV', 'HALLUCEVR', 'INHALEVER', 'METHAMEVR', 'PNRNMLIF', 'TRQNMLIF',
                         'STMANYLIF', 'SEDANYLIF', 'CIGFLAG', 'ALCFLAG'])


if __name__ == "__main__":
    main()
