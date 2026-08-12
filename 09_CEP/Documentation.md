
# COVID-19 Data Analysis – Singapore

## 1. Task Description

The goal of this assignment is to analyze and visualize COVID-19 data from **Singapore**.

The analysis focuses on several important pandemic indicators:

* 7-day COVID-19 incidence
* Hospitalization rate
* Criticality rate / ICU usage
* Vaccination progress
* Daily death rate
* Imported versus local COVID-19 cases

The project demonstrates basic data analysis techniques using **pandas**, numerical operations with **NumPy**, and data visualization with **Matplotlib**.

---

## 2. Pipeline Overview

```text
Load Data
    ↓
Convert and Sort Dates
    ↓
Calculate COVID-19 Indicators
    ↓
Handle Division by Zero
    ↓
Visualize Trends Over Time
```

---

## 3. Step 1: Import Libraries

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
```

### Explanation

| Library             | Purpose                                          |
| ------------------- | ------------------------------------------------ |
| `pandas`            | Loading, processing, and analyzing the dataset   |
| `numpy`             | Numerical operations and handling missing values |
| `matplotlib.pyplot` | Creating visualizations and line charts          |

---

## 4. Step 2: Load the Dataset

```python
df = pd.read_csv("data/covid19_sg.csv")
```

### Explanation

The COVID-19 dataset is loaded from a CSV file using `pandas`.

The dataset contains daily COVID-19 information for Singapore, including:

* Daily confirmed cases
* Imported cases
* Local transmissions
* Hospitalized patients
* ICU patients
* Vaccination data
* Daily deaths

---

## 5. Step 3: Convert and Sort the Date Column

```python
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date")
```

### Explanation

The `Date` column is converted into a proper datetime format.

```python
pd.to_datetime()
```

allows dates to be processed correctly as time-based values.

The dataset is then sorted chronologically:

```python
df = df.sort_values("Date")
```

This is important because calculations such as rolling averages and time-series visualizations require the data to be in the correct chronological order.

---

# 6. Step 4: Calculate the 7-Day Incidence

```python
population = 5686000

df["7_day_incidence"] = (
    df["Daily Confirmed"]
        .rolling(window=7)
        .sum()
        * 100000
        / population
)
```

### Explanation

The 7-day incidence measures the number of confirmed COVID-19 cases over the previous seven days per **100,000 inhabitants**.

The calculation follows this formula:

[
\text{7-Day Incidence}
======================

\frac{\text{Confirmed Cases in the Last 7 Days}}
{\text{Population}}
\times 100,000
]

The rolling window is created using:

```python
.rolling(window=7)
```

This creates a moving window containing the current day and the previous six days.

The number of confirmed cases within the window is calculated using:

```python
.sum()
```

The population of Singapore is defined as:

```python
population = 5686000
```

---

## 7. Visualizing the 7-Day Incidence

```python
plt.figure(figsize=(14,6))

plt.plot(df["Date"], df["7_day_incidence"], color="blue")

plt.title("7-Day Incidence")
plt.xlabel("Date")
plt.ylabel("Cases per 100,000 inhabitants")

plt.grid(True)

plt.show()
```

### Explanation

A line chart is used to visualize how the 7-day incidence changes over time.

The visualization shows:

* The development of COVID-19 case intensity
* Periods of increasing or decreasing infections
* Major infection waves

Using a 7-day rolling window also reduces daily fluctuations and makes general trends easier to identify.

---

# 8. Step 5: Calculate the Hospitalization Rate

```python
cases = df["Daily Confirmed"].replace(0, np.nan)
hospitalized = df["Still Hospitalised"].replace(0, np.nan)

df["hospitalization_rate"] = (
    hospitalized / cases
) * 100
```

### Explanation

Before calculating the hospitalization rate, zero values are replaced with `NaN`.

```python
.replace(0, np.nan)
```

This avoids division-by-zero errors.

The hospitalization rate is calculated as:

[
\text{Hospitalization Rate}
===========================

\frac{\text{Hospitalized Patients}}
{\text{Confirmed Cases}}
\times 100
]

The result represents the percentage relationship between hospitalized patients and confirmed COVID-19 cases.

---

## 9. Visualizing the Hospitalization Rate

```python
plt.figure(figsize=(12, 5))

plt.plot(df["Date"], df["hospitalization_rate"], color="orange")

plt.title("Hospitalization Rate")
plt.xlabel("Date")
plt.ylabel("Hospitalization Rate (%)")

plt.grid(True)

plt.show()
```

### Explanation

This visualization shows how the hospitalization rate changes over time.

It can help identify periods where a larger proportion of recorded COVID-19 cases was associated with hospitalization.

---

# 10. Step 6: Calculate Criticality

```python
df["criticality"] = (
    df["Intensive Care Unit (ICU)"] / hospitalized
) * 100
```

### Explanation

The criticality rate represents the percentage of hospitalized patients requiring treatment in an Intensive Care Unit (ICU).

The calculation is:

[
\text{Criticality}
==================

\frac{\text{ICU Patients}}
{\text{Hospitalized Patients}}
\times 100
]

This indicator provides information about the severity of cases among hospitalized patients.

---

## 11. Visualizing Criticality

```python
plt.figure(figsize=(12, 5))

plt.plot(df["Date"], df["criticality"], color="red")

plt.title("Criticality")
plt.xlabel("Date")
plt.ylabel("Hospitalized (%)")

plt.grid(True)

plt.show()
```

### Explanation

The chart visualizes how the percentage of hospitalized patients requiring ICU treatment changes over time.

This can be useful for observing changes in the severity of the pandemic and the pressure on intensive care resources.

---

# 12. Step 7: Visualize Vaccination Progress

```python
plt.figure(figsize=(14,6))

plt.plot(
    df["Date"],
    df["Cumulative Individuals Vaccination Completed"],
    color="green",
    linewidth=2
)

plt.title("Cumulative Individuals Fully Vaccinated")
plt.xlabel("Date")
plt.ylabel("Number of Individuals")

plt.grid(alpha=0.3)
plt.tight_layout()

plt.show()
```

### Explanation

This visualization shows the cumulative number of individuals who completed their COVID-19 vaccination.

Because the data is cumulative, the number generally increases over time.

The graph can be used to observe:

* The beginning of the vaccination campaign
* Periods of rapid vaccination
* Changes in vaccination growth over time

---

# 13. Step 8: Calculate the Daily Death Rate

```python
df["death_rate"] = (
    df["Daily Deaths"] /
    df["Daily Confirmed"].replace(0, np.nan)
) * 100
```

### Explanation

The daily death rate is calculated by dividing the number of daily deaths by the number of daily confirmed cases.

[
\text{Death Rate}
=================

\frac{\text{Daily Deaths}}
{\text{Daily Confirmed Cases}}
\times 100
]

Again, zero confirmed cases are replaced with `NaN` to avoid division by zero.

---

## 14. Visualizing the Daily Death Rate

```python
plt.figure(figsize=(14,6))

plt.plot(
    df["Date"],
    df["death_rate"],
    color="black"
)

plt.title("Daily COVID Death Rate")
plt.ylabel("Death Rate (%)")

plt.grid(alpha=0.3)

plt.show()
```

### Explanation

The chart shows changes in the calculated daily death rate over time.

This allows periods with relatively higher or lower death rates to be identified.

---

# 15. Step 9: Compare Imported and Local Cases

```python
plt.figure(figsize=(14,6))

plt.plot(
    df["Date"],
    df["Daily Imported"],
    label="Imported"
)

plt.plot(
    df["Date"],
    df["Daily Local transmission"],
    label="Local"
)

plt.title("Imported vs Local COVID Cases")

plt.legend()
plt.grid(alpha=0.3)

plt.show()
```

### Explanation

This visualization compares:

* **Imported cases:** COVID-19 cases associated with infections originating outside Singapore
* **Local transmission:** COVID-19 cases resulting from transmission within Singapore

The comparison helps illustrate how the pandemic developed over time.

For example, the visualization can show whether the number of local cases became more significant compared with imported cases.

---

# 16. Key Concepts

| Concept              | Explanation                                                               |
| -------------------- | ------------------------------------------------------------------------- |
| Time Series          | Data observed and analyzed over time                                      |
| `datetime`           | Data type used for working with dates and times                           |
| Sorting              | Arranging the dataset chronologically                                     |
| Rolling Window       | Calculation based on a moving group of observations                       |
| 7-Day Incidence      | Number of cases over seven days per 100,000 inhabitants                   |
| Hospitalization Rate | Percentage relationship between hospitalized patients and confirmed cases |
| Criticality          | Percentage of hospitalized patients requiring ICU treatment               |
| Cumulative Data      | Values that accumulate over time                                          |
| Data Visualization   | Presenting data and trends using charts                                   |
| `NaN`                | Represents a missing or undefined numerical value                         |
| Division by Zero     | Avoided by replacing zero values with `NaN`                               |

---

# 17. Discussion

This project demonstrates how a COVID-19 dataset can be transformed into several meaningful indicators and visualized as time series.

The analysis includes both **absolute values**, such as the cumulative number of vaccinated individuals, and **derived indicators**, such as:

* 7-day incidence
* Hospitalization rate
* Criticality
* Death rate

Using rolling calculations makes it possible to analyze infection trends over several days instead of focusing only on individual daily fluctuations.

The comparison between imported and local cases also provides additional insight into how the pandemic developed from external cases to community transmission.

---

# 18. Conclusion

This assignment demonstrates a basic time-series analysis workflow using Python.

The project shows how to:

* Load data from a CSV file
* Convert and process date values
* Sort time-series data chronologically
* Calculate rolling indicators
* Normalize case numbers relative to population
* Calculate percentage-based health indicators
* Handle division-by-zero problems using `NaN`
* Visualize multiple trends using Matplotlib

Overall, the project demonstrates how **pandas, NumPy, and Matplotlib** can be combined to analyze and visualize real-world health data.

