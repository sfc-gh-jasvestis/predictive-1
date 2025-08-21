USE DATABASE PM_DEMO;
USE SCHEMA PM_SCHEMA;
USE WAREHOUSE PM_WH;

CREATE OR REPLACE PROCEDURE SEED_DATA(NUM_MACHINES INTEGER, NUM_DAYS INTEGER, SEED INTEGER)
RETURNS STRING
LANGUAGE PYTHON
RUNTIME_VERSION = 3.10
PACKAGES = ('snowflake-snowpark-python', 'pandas', 'numpy')
HANDLER = 'run'
AS
$$
from snowflake.snowpark.session import Session
from snowflake.snowpark import functions as F
import pandas as pd
import numpy as np
from datetime import date, timedelta


def generate_weather(num_days: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed + 17)
    end = pd.to_datetime(date.today())
    dates = pd.date_range(end=end, periods=num_days, freq='D')
    day_idx = np.arange(num_days)
    ambient_temp = 18 + 10 * np.sin(2 * np.pi * day_idx / 365) + rng.normal(0, 2, num_days)
    humidity = 55 + 20 * np.sin(2 * np.pi * (day_idx + 90) / 365) + rng.normal(0, 5, num_days)
    humidity = np.clip(humidity, 15, 100)
    rain_prob = 0.25 + 0.15 * np.sin(2 * np.pi * (day_idx + 45) / 365)
    rainfall_mm = (rng.random(num_days) < np.clip(rain_prob, 0.05, 0.6)) * rng.gamma(2.0, 3.0, num_days)
    return pd.DataFrame({
        'date': dates,
        'ambient_temp': ambient_temp,
        'humidity': humidity,
        'rainfall_mm': rainfall_mm,
    })


def simulate_machine_series(machine_id: int, num_days: int, weather: pd.DataFrame, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed + 1000 + machine_id)
    dates = weather['date'].values
    age_days = np.arange(num_days) + rng.integers(0, 365)

    weekly = 8 + 2 * np.sin(2 * np.pi * (np.arange(num_days) % 7) / 7)
    usage_hours = weekly + rng.normal(0, 1.0, num_days)
    usage_hours = np.clip(usage_hours, 0, 24)

    internal_temp = weather['ambient_temp'].values + 0.9 * usage_hours + rng.normal(0, 1.5, num_days)

    vibration = 0.06 * usage_hours + 0.0008 * age_days + 0.01 * (weather['humidity'].values - 50)
    vibration += rng.normal(0, 0.05, num_days)

    z = (
        -6.0
        + 0.25 * (usage_hours - 8)
        + 0.05 * (internal_temp - 35)
        + 8.0 * np.maximum(vibration - 1.0, 0.0)
        + 0.012 * (age_days - 200)
        + 0.012 * (weather['humidity'].values - 60)
        + 0.03 * (weather['rainfall_mm'].values > 0).astype(float)
    )
    p_fail = 1 / (1 + np.exp(-z))
    breakdown = (rng.random(num_days) < np.clip(p_fail, 0.0, 0.85)).astype(int)

    df = pd.DataFrame({
        'date': dates,
        'machine_id': machine_id,
        'age_days': age_days,
        'usage_hours': usage_hours,
        'internal_temp': internal_temp,
        'vibration': vibration,
        'ambient_temp': weather['ambient_temp'].values,
        'humidity': weather['humidity'].values,
        'rainfall_mm': weather['rainfall_mm'].values,
        'breakdown': breakdown,
    })

    df['serviced'] = 0
    for i in range(1, num_days):
        if df.loc[i - 1, 'breakdown'] == 1:
            df.loc[i, 'serviced'] = 1
            df.loc[i:, 'age_days'] = df.loc[i:, 'age_days'].values - 90
            df.loc[i:, 'vibration'] = df.loc[i:, 'vibration'].values - 0.15
    df['age_days'] = np.maximum(df['age_days'], 0)
    df['vibration'] = np.maximum(df['vibration'], 0.0)
    return df


def run(session: Session, NUM_MACHINES: int, NUM_DAYS: int, SEED: int) -> str:
    weather = generate_weather(NUM_DAYS, SEED)

    machines = pd.DataFrame({
        'machine_id': np.arange(1, NUM_MACHINES + 1),
        'model': [f'M-{100 + i}' for i in range(NUM_MACHINES)],
        'installed_on': pd.to_datetime(date.today()) - pd.to_timedelta(np.random.default_rng(SEED).integers(200, 2000, NUM_MACHINES), unit='D'),
    })

    series = []
    for mid in machines['machine_id']:
        series.append(simulate_machine_series(int(mid), NUM_DAYS, weather, SEED))
    daily = pd.concat(series, ignore_index=True)

    # Persist tables using Snowpark (avoids external dependencies like pyarrow)
    session.create_dataframe(weather).write.mode("overwrite").save_as_table("WEATHER_DAILY")
    session.create_dataframe(machines).write.mode("overwrite").save_as_table("MACHINES")
    session.create_dataframe(daily).write.mode("overwrite").save_as_table("MACHINE_DAILY")

    return f"Seeded WEATHER_DAILY={len(weather)}, MACHINES={len(machines)}, MACHINE_DAILY={len(daily)} rows."
$$;

-- Seed example
-- CALL SEED_DATA(10, 365, 42);
