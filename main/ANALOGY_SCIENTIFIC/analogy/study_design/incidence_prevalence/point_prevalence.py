from typing import Tuple
from datetime import datetime
import pandas as pd

"""
File consist of functions to calculate point prevalence.
Below is the list of definitions used in the code.

1. PRACTICE_PATIENT_ID -> The unique key that identifies a single patient.

2. START_DATE -> This is the patient start date. It is defined as the latest of:
                    (a) Vision or Computerization date
                    (b) Acceptable Mortality Reporting(AMR) date
                    (c) Patient registration date
                    (d) Study start date
                    (e) Date on which patient becomes eligible for the study based on age restrictions(if any).

3. END_DATE -> This is the patient end date. It is defined as the earliest of:
                    (a) Practice collection date
                    (b) Patient transfer date
                    (c) Patient death date
                    (d) Study end date
                    (e) Maximum age until which patient is eligible to participate in the study.

4. INDEX_DATE -> Date on which the patient follow-up starts.

5. REGISTRATION_STATUS: This indicates the registration status of the patient to the practice.

6. TRANSFER_DATE -> Date on which the patient transferred out of the practice.

7. DEATH_DATE -> Date on which the patient has died.

8. COLLECTION_DATE -> This was the last time data was collected from the practice.

* The Date/Time in DExtER works with the ISO 8601[4] format, which is (yyyy-MM-dd)
"""


def point_prevalence(dataframe: pd.DataFrame = None,
                     study_start: datetime = None,
                     colname: str = None) -> Tuple:
    """
    Function definition for point prevalence calculation.

    Args:
        dataframe: the full or grouped pandas dataframe.
        study_start: year value to calculate point prevalence.
        colname: baseline variable column name.

    Return:
        tuple (year, point_prevalence, denominator, numerator, lower_ci, upper_ci, error_delta)

    """
    # study_start date is 1st of Jan, study_year : midnight (00:00:00)
    study_start = datetime(year=study_year, month=1, day=1, hour=0, minute=0, second=0)

    # event that occured before the year of interest which is a combination of outcome recording that is
    # before that year of interest
    numerator = len(dataframe[((dataframe['INDEX_DATE'] <= study_start) &
                               (dataframe['END_DATE'] >= study_start)) &
                              (dataframe[datecol_name] <= study_start)]['PRACTICE_PATIENT_ID'])

    # Patients who are in the practice at the start of the interested year, that is they enter the cohort
    # before the start of the interested year and have not exited before the start of the interested year
    denominator = len(dataframe[(dataframe['INDEX_DATE'] <= study_start) &
                                (dataframe['END_DATE'] >= study_start)]['PRACTICE_PATIENT_ID'])

    point_prev = (numerator / (denominator + SMALL_FP_VAL))  # adding a small constant to avoid division by zero.

    error = (CI_CONSTANT * (np.sqrt((point_prev * (1 - point_prev)) / denominator)))

    lower_ci = point_prev - error
    upper_ci = point_prev + error

    return (
    study_year, point_prev * PER_PY, denominator, numerator, lower_ci * PER_PY, upper_ci * PER_PY, error * PER_PY)