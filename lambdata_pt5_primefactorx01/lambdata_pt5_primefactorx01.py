import pandas as pd
import numpy as np
from sklearn import metrics


class Helper:

    STATES = {
        "AL": "Alabama",
        "AK": "Alaska",
        "AS": "American Samoa",
        "AZ": "Arizona",
        "AR": "Arkansas",
        "CA": "California",
        "CO": "Colorado",
        "CT": "Connecticut",
        "DE": "Delaware",
        "DC": "District Of Columbia",
        "FM": "Federated States Of Micronesia",
        "FL": "Florida",
        "GA": "Georgia",
        "GU": "Guam",
        "HI": "Hawaii",
        "ID": "Idaho",
        "IL": "Illinois",
        "IN": "Indiana",
        "IA": "Iowa",
        "KS": "Kansas",
        "KY": "Kentucky",
        "LA": "Louisiana",
        "ME": "Maine",
        "MH": "Marshall Islands",
        "MD": "Maryland",
        "MA": "Massachusetts",
        "MI": "Michigan",
        "MN": "Minnesota",
        "MS": "Mississippi",
        "MO": "Missouri",
        "MT": "Montana",
        "NE": "Nebraska",
        "NV": "Nevada",
        "NH": "New Hampshire",
        "NJ": "New Jersey",
        "NM": "New Mexico",
        "NY": "New York",
        "NC": "North Carolina",
        "ND": "North Dakota",
        "MP": "Northern Mariana Islands",
        "OH": "Ohio",
        "OK": "Oklahoma",
        "OR": "Oregon",
        "PW": "Palau",
        "PA": "Pennsylvania",
        "PR": "Puerto Rico",
        "RI": "Rhode Island",
        "SC": "South Carolina",
        "SD": "South Dakota",
        "TN": "Tennessee",
        "TX": "Texas",
        "UT": "Utah",
        "VT": "Vermont",
        "VI": "Virgin Islands",
        "VA": "Virginia",
        "WA": "Washington",
        "WV": "West Virginia",
        "WI": "Wisconsin",
        "WY": "Wyoming",
    }

    @staticmethod
    def print_nulls(df: pd.DataFrame):
        """Prints the number of nulls for each column"""

        number_of_nulls = df.isna().sum()
        print(f"Nulls:\n{number_of_nulls}")

    @staticmethod
    def predict(X: pd.DataFrame):
        """Returns true or false predictions for each row ins a dataframe"""

        length = X.shape[0]
        return np.random.choice([True, False], length)

    @staticmethod
    def print_confusion_matrix(y, y_pred):
        """ Prints a confusion matrix with"""

        df = pd.DataFrame(
            metrics.confusion_matrix(y, y_pred),
            columns=["positive", "negative"],
            index=["True", "False"],
        )
        print(df)

    @staticmethod
    def convert_states(s: pd.Series):
        """ Convert state abbreviations to full names"""
        """ Converts df['state'] from abbrev to full"""

        return s.map(Helper.STATES)


if __name__ == "__main__":
    X, y = np.arange(10).reshape((5, 2)), [True, False, True, False, True]
    pred = Helper.predict(X)
    Helper.print_confusion_matrix(y, pred)
    states = pd.Series(["CA", "TX", "ND"])

    print(Helper.convert_states(states))
