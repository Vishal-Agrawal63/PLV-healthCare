# Guide to Entering Patient Data

This form is designed to capture the key characteristics of a patient that the machine learning model uses to predict their annual healthcare expenditure. Here’s what each field means:

---

### 1. Age
*   **What it is:** The patient's age in years.
*   **How to enter:** Type a whole number.
*   **Example:** `45`

### 2. Sex
*   **What it is:** The patient's biological sex.
*   **How to enter:** Select an option from the dropdown menu. The application automatically converts this to the required numeric code for the model (`1` for Male, `2` for Female).
*   **Example:** Select `Male` from the dropdown.

### 3. Race (Code)
*   **What it is:** A numeric code representing the patient's race and ethnicity, based on the standard codes used in the MEPS dataset.
*   **How to enter:** You need to enter the specific number code that corresponds to a category. Since the model was trained on these codes, using the correct one is important. Common codes from this dataset are:
    *   `1`: Hispanic
    *   `2`: Non-Hispanic White
    *   `3`: Non-Hispanic Black
    *   `4`: Non-Hispanic Asian
    *   `5`: Non-Hispanic Other/Multiple Race
*   **Example:** To represent a non-Hispanic White patient, enter `2`.

### 4. Poverty Category (1-5)
*   **What it is:** The patient's household income level relative to the U.S. federal poverty line.
*   **How to enter:** Enter a number from 1 to 5 based on this scale:
    *   `1`: Poor / Negative income
    *   `2`: Near Poor (Just above the poverty line)
    *   `3`: Low Income
    *   `4`: Middle Income
    *   `5`: High Income
*   **Example:** `4` represents a middle-income patient.

### 5. Insurance Coverage (1-3)
*   **What it is:** The patient's primary insurance status for the year.
*   **How to enter:** Enter a number from 1 to 3:
    *   `1`: Has private insurance coverage.
    *   `2`: Has public insurance coverage (like Medicare or Medicaid).
    *   `3`: Is uninsured.
*   **Example:** `1` indicates the patient has private insurance.

### 6. Health Status (1-5)
*   **What it is:** The patient's self-reported general health status.
*   **How to enter:** Enter a number from 1 to 5, where a lower number means better health:
    *   `1`: Excellent
    *   `2`: Very Good
    *   `3`: Good
    *   `4`: Fair
    *   `5`: Poor
*   **Example:** `3` represents a patient in "Good" health.

### 7. Annual Office Visits
*   **What it is:** The total number of times the patient visited a doctor's office or clinic in a year.
*   **How to enter:** Type a whole number (e.g., `0`, `1`, `2`, ...).
*   **Example:** `5` means the patient had five office visits last year.

### 8. Annual Outpatient Visits
*   **What it is:** The total number of visits to a hospital's outpatient department in a year. This is for care that doesn't require an overnight stay.
*   **How to enter:** Type a whole number.
*   **Example:** `1` means one visit to a hospital outpatient clinic.

### 9. Annual ER Visits
*   **What it is:** The total number of times the patient went to the emergency room in a year.
*   **How to enter:** Type a whole number.
*   **Example:** `0` means the patient had no emergency room visits.

### 10. Annual Hospital Discharges
*   **What it is:** The number of times the patient was admitted to a hospital for an overnight stay and was later discharged. This is essentially a count of their hospital stays during the year.
*   **How to enter:** Type a whole number.
*   **Example:** `0` means the patient was not hospitalized overnight during the year.