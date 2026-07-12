# Changelog
All notable changes to this project will be documented in this file.

---
## v2.2.1 - Small Modifications for Better Visualizations
- Cleaner visualizations in `.sql` queries;
- PT-BR titles and descriptions in `04_sql.ipynb`;
- Several tiny modifications in pt-br subtitles in notebooks 01, 02 and 03.

---
## v2.2.0 - Applying SQL
- Data infrastructure simulation through the normalization of raw data into 3 relational tables hosted on a cloud PostgreSQL database (Neon.tech);
- Data extraction, ingest and analyses through secure data loading and recovery and some tabular analyses using SQL queries (`LEFT JOINs`, aggregates, CTEs);
- PostgreSQL integration via `SQLAlchemy` and `psycopg2` as the database driver;
- Database credentials securely isolated via Environment Variables (`.env`) and protected through `.gitignore` protocols;

---
## v2.1.0 - XGBoost
- Added the XGBoost model.

---
## v2.0.0 - Transforming the Project into a Installable Python Package
- Discontinued the use of `requirements.txt` for external dependencies;
- Adopted a modern python packaging convention which combines the installation of external dependencies and the locally editable source code as a package using `pyproject.toml`;
- Restructured the `src/` to match the new convention adopted.

---
## v1.1.0 - Documentation and Internationalization Update
- Added english README version;
- Enhanced functions to plot figures and display tables and texts in english;
- Added english notebook plots and figures;
- Added english project documentation.

---
## v1.0.0 - Initial Complete Release
- Exploratory Data Analysis (EDA);
- Machine Learning pipeline;
- Hyperparameter optimization;
- Threshold analysis for business-oriented decisions;
- Modular structure and documentation.