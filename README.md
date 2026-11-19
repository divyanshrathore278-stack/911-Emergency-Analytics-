# 🚨 911 Emergency Call Analytics
### From Emergency Call Patterns to Predictive Insights & Resource Optimization

An end-to-end Data Science and Machine Learning project focused on analyzing emergency 911 calls, identifying geographical hotspots, forecasting future call demand, and developing data-driven emergency staffing recommendations.

---

## 📌 Project Overview

Emergency response systems must make critical decisions about resource allocation, staffing, and geographical coverage.

This project explores an important operational question:

**How should an emergency dispatch center allocate its resources based on when, where, and what types of emergency calls are received?**

Using the Montgomery County 911 Calls dataset, this project builds a complete analytical pipeline that transforms raw emergency call records into actionable insights.

Rather than limiting the analysis to descriptive statistics and visualizations, the project combines:

- Exploratory Data Analysis (EDA)
- Statistical Hypothesis Testing
- Geospatial Clustering and Hotspot Detection
- Time-Series Forecasting
- Machine Learning
- Emergency Staffing Optimization
- Interactive Data Visualization

The ultimate goal is to demonstrate how Data Science can support better emergency response planning and operational decision-making.

---

## 🎯 Project Objectives

The project focuses on six major objectives:

1. **Understand emergency call patterns:** Identify the most frequent emergency categories, common reasons, and temporal patterns.

2. **Discover statistically significant differences:** Investigate whether emergency call volumes differ between weekdays and weekends and whether emergency categories vary across townships.

3. **Identify geographical hotspots:** Discover areas with concentrated emergency call activity using spatial clustering.

4. **Forecast emergency demand:** Predict hourly emergency call volumes using machine learning and compare model performance against a seasonal-naive baseline.

5. **Optimize emergency resource allocation:** Estimate the number of emergency units required during different hours and weekdays.

6. **Visualize analytical results:** Generate interactive geographical maps, statistical reports, and an interactive Streamlit dashboard.

---

## 📊 Dataset

**Dataset:** Emergency - 911 Calls

**Source:** [Montgomery County 911 Calls Dataset – Kaggle](https://www.kaggle.com/datasets/mchirico/montcoalert)

**Location:** Montgomery County, Pennsylvania, USA.

The dataset contains emergency call records with information such as:

| Feature | Description |
|---|---|
| `lat` | Latitude of the emergency call |
| `lng` | Longitude of the emergency call |
| `desc` | Description of the emergency call |
| `zip` | ZIP code |
| `title` | Emergency category and reason |
| `timeStamp` | Date and time of the call |
| `twp` | Township associated with the call |
| `addr` | Address of the emergency call |
| `e` | Dataset field included in the original source |

### Data Preprocessing

The preprocessing pipeline includes:

- Removing duplicate records.
- Converting timestamps into datetime format.
- Handling missing values.
- Removing invalid geographical coordinates using a Montgomery County bounding box.
- Extracting emergency categories and reasons from the `title` column.
- Creating temporal features such as hour, weekday, month, and date.
- Identifying weekend versus weekday calls.
- Sorting records chronologically.
- Saving the cleaned dataset in Parquet format.

The processed dataset is used by the subsequent analytical modules.

---

## 🧠 Project Architecture

The project follows a modular architecture, separating data preparation, statistical analysis, machine learning, geographical analysis, and visualization.

### 1. Exploratory Data Analysis (EDA)

**Module:** `src/eda.py`

The EDA module explores the distribution and temporal behavior of emergency calls.

Analysis includes:

- Emergency call distribution by category.
- Most frequent emergency reasons.
- Average hourly call volume by weekday.
- Hour-versus-weekday heatmaps.
- Monthly emergency call trends.
- Townships with the highest call volumes.

**Key output:** Statistical summaries and visualizations that help identify demand patterns and peak emergency periods.

### 2. Statistical Hypothesis Testing

**Module:** `src/stats_tests.py`

Statistical testing is used to investigate whether observed differences in emergency call patterns are meaningful.

**Mann–Whitney U Test**

Used to compare daily emergency call volumes between weekdays and weekends.

Additional measurements:
- Mean daily calls.
- Percentage difference.
- P-value.
- Rank-biserial correlation.
- Cohen's d.

**Chi-Square Test of Independence**

Used to investigate whether emergency category distributions differ across the ten busiest townships.

Additional measurement:
- Chi-square statistic.
- Degrees of freedom.
- P-value.
- Cramér's V effect size.

The project considers effect sizes alongside p-values to provide a better understanding of the practical importance of observed differences.

### 3. Geospatial Analysis & Hotspot Detection

**Module:** `src/geo.py`

Geospatial analysis identifies geographical concentrations of emergency calls and explores potential emergency station locations.

**DBSCAN Clustering**

Density-Based Spatial Clustering of Applications with Noise is used to identify geographical hotspots.

Implementation details:
- Haversine distance metric.
- 0.5 km neighborhood radius.
- Minimum 40 samples per cluster.
- Maximum sample size of 60,000 EMS calls.

The detected clusters are ranked using estimated call counts scaled from the analyzed sample.

**K-Means Clustering**

K-Means is used to explore potential emergency station locations.

The model evaluates between 2 and 12 hypothetical station locations.

A geographical coverage curve measures the average distance between an emergency call and its nearest proposed station.

**Folium Interactive Mapping**

The geographical analysis generates an interactive map containing:
- Emergency call density heatmaps.
- Detected geographical hotspots.
- Proposed emergency station locations.

### 4. Machine Learning & Demand Forecasting

**Module:** `src/forecast.py`

A machine learning model is developed to predict hourly emergency call volumes.

**Algorithm:** HistGradientBoostingRegressor

**Loss Function:** Poisson

The forecasting model uses engineered time-series features, including:

| Feature | Description |
|---|---|
| `lag_24` | Call volume 24 hours earlier |
| `lag_48` | Call volume 48 hours earlier |
| `lag_168` | Call volume at the same hour one week earlier |
| `roll_168_mean` | Previous 168-hour rolling average, shifted to avoid leakage |
| `hour` | Hour of the day |
| `dow` | Day of the week |
| `month` | Month |
| `is_weekend` | Weekend indicator |

**Model Evaluation**

The project compares machine learning performance against a seasonal-naive baseline that uses the call volume from the same hour during the previous week.

Evaluation methods:

- Chronological holdout testing.
- Rolling-origin time-series cross-validation.
- Mean Absolute Error (MAE).
- Weighted Absolute Percentage Error (WAPE).

The final holdout contains approximately the last 60 days of eligible hourly observations.

Features are constructed so that the historical call-volume information used for each prediction is at least 24 hours older than its target.

### 5. Emergency Staffing Recommendations

**Module:** `src/staffing.py`

This module converts historical emergency demand patterns into estimated staffing requirements.

The model calculates the 90th percentile (P90) of hourly call demand for each emergency category, weekday, and hour.

The staffing calculation is:

\[
Units\ Required =
\left\lceil
\frac{P90\ Demand \times Handling\ Time}
{Target\ Utilization}
\right\rceil
\]

**Planning assumptions:**

| Emergency Category | Assumed Handling Time |
|---|---:|
| EMS | 1.00 hour |
| Fire | 1.25 hours |
| Traffic | 0.75 hours |

Target unit utilization: 60%.

These are configurable planning assumptions, not measured emergency-service operational statistics.

The module generates:
- Staffing recommendation CSV.
- Category-specific staffing heatmaps.
- Hourly and weekday resource requirements.

### 6. Interactive Dashboard

**Module:** `app/dashboard.py`

The project includes an interactive dashboard developed using Streamlit.

Dashboard functionality:

- Filter emergency calls by category.
- Select a date range.
- View total emergency calls.
- View average daily emergency calls.
- Identify the busiest hour.
- Explore hourly and weekday demand patterns.
- Visualize monthly call trends.
- Explore geographical call distributions.

The dashboard uses the processed dataset generated by the main pipeline.

---

## 🛠️ Technology Stack

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Data Manipulation | Pandas, NumPy |
| Data Visualization | Matplotlib, Seaborn |
| Statistical Analysis | SciPy |
| Machine Learning | Scikit-learn |
| Geospatial Analysis | Folium |
| Spatial Clustering | DBSCAN, K-Means |
| Forecasting | HistGradientBoostingRegressor |
| Interactive Dashboard | Streamlit |
| Data Storage | CSV, Parquet |
| Version Control | Git, GitHub |

---

## 📁 Project Structure

```text
emergency-911-analytics/
│
├── app/
│   └── dashboard.py
│
├── data/
│   ├── 911.csv
│   └── processed/
│
├── reports/
│   ├── figures/
│   ├── hotspot_map.html
│   ├── ems_hotspots.csv
│   ├── staffing_recommendation.csv
│   ├── forecast_holdout_metrics.csv
│   ├── forecast_cv_metrics.csv
│   ├── station_coverage_curve.csv
│   ├── stats_results.json
│   └── key_findings.md
│
├── scripts/
│   └── make_sample_data.py
│
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data.py
│   ├── eda.py
│   ├── stats_tests.py
│   ├── geo.py
│   ├── forecast.py
│   └── staffing.py
│
├── run_pipeline.py
├── requirements.txt
├── .gitignore
└── README.md
```

---

## ⚙️ Installation & Setup

Follow these instructions to run the project locally.

### Step 1: Clone the Repository

```bash
git clone https://github.com/YOUR-USERNAME/emergency-911-analytics.git
```

Navigate into the project directory:

```bash
cd emergency-911-analytics
```

### Step 2: Create a Virtual Environment

```bash
python -m venv venv
```

Activate the environment.

**Windows:**

```bash
venv\Scripts\activate
```

**macOS / Linux:**

```bash
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Download the Dataset

Download the original 911 emergency calls dataset from Kaggle:

[Download Dataset](https://www.kaggle.com/datasets/mchirico/montcoalert)

Place the downloaded `911.csv` file inside the `data/` directory.

Expected location:

```text
data/911.csv
```

### Step 5: Run the Complete Analytical Pipeline

```bash
python run_pipeline.py --data data/911.csv
```

This command executes the complete workflow:

1. Data loading and preprocessing.
2. Exploratory data analysis.
3. Statistical hypothesis testing.
4. Geospatial hotspot detection.
5. Machine learning forecasting.
6. Staffing recommendations.
7. Automatic generation of key findings.

### Step 6: Launch the Interactive Dashboard

After running the pipeline:

```bash
streamlit run app/dashboard.py
```

Streamlit will provide a local URL where you can interact with the dashboard.

---

## 🧪 Testing Without the Original Dataset

The project includes a synthetic dataset generator.

Run:

```bash
python scripts/make_sample_data.py
```

This generates:

```text
data/911_sample.csv
```

Run the pipeline using the synthetic dataset:

```bash
python run_pipeline.py --data data/911_sample.csv
```

**Important:** The generated dataset is artificial and is intended only for testing the pipeline. Its analytical results must not be presented as real-world emergency call findings.

---

## 📈 Generated Outputs

The pipeline automatically generates analytical reports inside the `reports/` directory.

| Output | Description |
|---|---|
| `01_category_mix.png` | Emergency category distribution |
| `02_top_reasons.png` | Most frequent emergency reasons |
| `03_heatmap_all.png` | Hourly and weekday call distribution |
| `04_monthly_trend.png` | Monthly emergency call trends |
| `05_top_townships.png` | Highest-volume townships |
| `06_station_coverage_curve.png` | Hypothetical station coverage analysis |
| `07_forecast_holdout.png` | Actual versus predicted hourly demand |
| `08_staffing_*.png` | Category-specific staffing heatmaps |
| `hotspot_map.html` | Interactive geographical hotspot map |
| `ems_hotspots.csv` | Detected EMS geographical hotspots |
| `staffing_recommendation.csv` | Estimated staffing requirements |
| `forecast_holdout_metrics.csv` | Holdout model evaluation |
| `forecast_cv_metrics.csv` | Time-series cross-validation results |
| `station_coverage_curve.csv` | Station coverage measurements |
| `stats_results.json` | Statistical test results |
| `key_findings.md` | Automatically generated analytical summary |

---

## ⚠️ Assumptions & Limitations

This project is designed for analytical exploration and educational purposes. Its results should not be interpreted as validated operational recommendations for an actual emergency dispatch center.

Important limitations:

**1. Dataset Coverage**

The analysis is based on emergency calls from Montgomery County, Pennsylvania. Findings may not generalize to other geographical regions.

**2. Staffing Assumptions**

Handling times and target utilization are assumed parameters. The dataset does not provide actual unit-level availability, incident duration, or operational response-time information.

**3. Geographical Station Placement**

K-Means uses geographical coordinates to estimate station locations. It does not account for road networks, traffic conditions, terrain, existing emergency stations, or real-world travel times.

**4. Hotspot Estimation**

DBSCAN is applied to a sample of EMS calls. Cluster call counts are scaled estimates rather than exact operational totals.

**5. Forecasting**

The forecasting model uses historical call patterns. It does not currently incorporate external variables such as weather conditions, public events, holidays, or sudden emergencies.

Its rolling evaluation is not equivalent to a single forecast of all future demand made in advance.

**6. Operational Deployment**

The project is an analytical prototype and has not been validated or deployed in a real emergency response system.

---

## 🚀 Future Improvements

Potential extensions to make the project more advanced:

- Compare HistGradientBoosting with XGBoost, LightGBM, and other forecasting approaches.
- Add SHAP-based model explainability.
- Incorporate weather and holiday information.
- Develop more advanced forecasting models.
- Introduce road-network-based travel-time optimization.
- Evaluate geographical hotspot stability across different time periods.
- Add real-time demand simulation.
- Develop an advanced emergency operations dashboard.
- Deploy the application using Streamlit Community Cloud or another hosting platform.
- Add automated data validation and model monitoring.

---

## 🎓 Learning Outcomes

Through this project, I explored and implemented several important Data Science concepts:

- End-to-end data preprocessing and feature engineering.
- Exploratory data analysis and statistical interpretation.
- Hypothesis testing and effect-size analysis.
- Density-based geographical clustering.
- K-Means clustering for hypothetical facility placement.
- Time-series feature engineering.
- Chronological model validation.
- Machine learning model evaluation.
- Data-driven resource planning.
- Interactive dashboard development.
- Modular Python project architecture.

---

## 👨‍💻 Author

**Divyansh Rathore**

B.Tech – Artificial Intelligence & Data Science

Interested in Data Science, Machine Learning, Predictive Analytics, and Artificial Intelligence.

📍 Indore, Madhya Pradesh, India

---

## ⭐ Support

If you find this project interesting, consider giving the repository a star!

Feedback, suggestions, and contributions are welcome.

**Turning emergency data into meaningful insights through Data Science.**# 911-Emergency-Analytics-
911 call analytics: statistical testing, geospatial hotspots, demand forecasting and staffing recommendations with a Streamlit dashboard.
