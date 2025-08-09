-- Set your role and account-specific settings as needed
-- USE ROLE ACCOUNTADMIN;

-- 1) Warehouse
CREATE OR REPLACE WAREHOUSE PM_WH
  WAREHOUSE_SIZE = XSMALL
  AUTO_SUSPEND = 60
  AUTO_RESUME = TRUE
  INITIALLY_SUSPENDED = TRUE;

-- 2) Database and Schema
CREATE OR REPLACE DATABASE PM_DEMO;
CREATE OR REPLACE SCHEMA PM_DEMO.PM_SCHEMA;
USE DATABASE PM_DEMO;
USE SCHEMA PM_SCHEMA;
USE WAREHOUSE PM_WH;

-- 3) Stage to store Streamlit files
CREATE OR REPLACE STAGE PM_STAGE;

-- 4) After uploading app files to the stage (e.g., app.py):
--    CREATE OR REPLACE STREAMLIT PM_APP
--      ROOT_LOCATION = '@PM_STAGE'
--      MAIN_FILE = 'streamlit_app.py'
--      QUERY_WAREHOUSE = PM_WH;

-- You can manage packages for the Streamlit app from Snowsight -> Streamlit app -> Packages.
