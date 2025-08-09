# Predictive Maintenance Streamlit Demo

This demo showcases how predictive maintenance can help prevent machine breakdowns using synthetic data, simple ML modeling, and interactive dashboards.

## Features
- Synthetic historical dataset for multiple machines with weather factors
- Dashboards for health trends, breakdowns, and KPIs
- ML model to predict failure risk within a future horizon
- Forward-looking recommendations: when to service each machine
- Weather effect analysis via feature importance and what-if controls

## Quickstart (Local)

```bash
# From project root
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Snowflake Version (Streamlit in Snowflake)

1) Create warehouse, database, schema, and stage
- Open a Snowflake SQL worksheet and run:
  - `snowflake/snowflake_setup.sql`

2) Create the seeding stored procedure and seed data
- Run the SQL in `snowflake/seed_data_sp.sql`
- Seed example:
  ```sql
  CALL SEED_DATA(10, 365, 42);
  ```

3) Upload Streamlit app to stage and create the app
- In Snowsight -> Databases -> `PM_DEMO` -> `PM_SCHEMA` -> Stages -> `PM_STAGE`
  - Upload `snowflake/streamlit_app.py`
- In a SQL worksheet, create the Streamlit app:
  ```sql
  USE DATABASE PM_DEMO;
  USE SCHEMA PM_SCHEMA;
  CREATE OR REPLACE STREAMLIT PM_APP
    ROOT_LOCATION = '@PM_STAGE'
    MAIN_FILE = 'streamlit_app.py'
    QUERY_WAREHOUSE = PM_WH;
  ```
- Open the app from Snowsight -> Streamlit -> `PM_APP`.

4) Packages
- In the Streamlit app settings (Packages), add: `snowflake-snowpark-python`, `pandas`, `numpy`, `scikit-learn`, `altair`.

5) Using the app
- Use the sidebar to optionally reseed data in Snowflake (calls `SEED_DATA`).
- Dashboards and model training run inside Streamlit in Snowflake, querying `MACHINE_DAILY`.

## Notes
- All data is simulated via a Snowflake stored procedure for the Snowflake version.
- The model is a simple Random Forest classifier trained on the retrieved dataset.
