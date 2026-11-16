# Step 1: Data model

Builds the Postgres database the CRM analytics assistant queries. The design and every
cleaning decision are in [`step1-data-model-olist.md`](../step1-data-model-olist.md).

- **`raw` schema:** the Olist CSVs loaded as they are, every column `text`. Never edited by hand.
- **`analytics` schema:** cleaned tables with real types, keys and a comment on every column.
  The LLM and the `crm_readonly` user only see this schema.

## Requirements

- Postgres 14 or newer with the `unaccent` extension (tested on Neon, Postgres 18).
- Python 3.11+ with the packages in [`requirements.txt`](../requirements.txt):
  `pip install -r requirements.txt`. No `psql` needed.
- A `.env` file in the project root (or the same variables in the environment):

  ```
  DATABASE_URL="postgresql://<user>:<password>@<host>/<database>?sslmode=require"
  READONLY_PASSWORD="<long random password for crm_readonly>"
  ```

  Never commit `.env`.

## Running the three steps

Run from the project root. On Windows, use `.venv\Scripts\python.exe` in place of `python`.

**1. Load raw**

```
python load_raw.py                  # reload all 11 raw tables
python load_raw.py products leads   # reload only the named tables
```

Reads the CSVs from `olist_public_dataset/` and `olist_marketing_dataset/`, drops and
recreates each `raw` table, loads it with `COPY` and prints its row count. Files are read as
`utf-8-sig`, so a byte-order mark isn't glued to the first header.

**2. Run the checks**

```
python db/run_sql.py db/checks.sql
```

Read-only queries against `raw` that confirm the assumptions behind the cleaning decisions.
Each check runs in its own read-only transaction and prints its label and result. If a
result contradicts its "expect" note in the spec, stop and review the decision before
transforming. (`checks.sql` needs `unaccent`, which `transform.sql` installs; on a brand-new
database run `CREATE EXTENSION IF NOT EXISTS unaccent;` first.)

**3. Build analytics**

```
python db/run_sql.py db/transform.sql
```

Drops and rebuilds the `analytics` schema in one transaction: setup, the ten tables in
build order (geolocation, orders, customers, products, sellers, order_items,
order_payments, order_reviews, leads, closed_deals), their keys and comments, then the
grants. If any statement fails, the runner prints it with the database error and rolls
back, leaving `analytics` as it was. On success it sets the `crm_readonly` password from
`READONLY_PASSWORD`. Safe to run again.

## Files

| File | Purpose |
| --- | --- |
| `../load_raw.py` | Loads every CSV into `raw` (all `text`) |
| `checks.sql` | Check queries against `raw`, each with an `\echo` label |
| `transform.sql` | Drops and rebuilds `analytics`, then grants `SELECT` to `crm_readonly` |
| `run_sql.py` | Runs the two `.sql` files with psycopg2 (they remain valid `psql` scripts) |

## Changes from the spec

Approved after reviewing the check results:

- **Sellers:** the city is also split on a backslash, and a city that is only a state code
  (`sp`) or the abbreviation `sbc` is replaced with geolocation's city for the zip.
- **Closed deals:** deals recorded as won before the lead's first contact are excluded
  (1 deal in the data). It stays in `raw`.
- **Products:** the 4 products with weight 0 g are kept as 0, as in the spec.

## The read-only user

`crm_readonly` can `SELECT` from `analytics` only. Its sessions default to
`search_path = analytics`, read-only transactions, a 5-second statement timeout and the
`America/Sao_Paulo` time zone. It has no access to `raw`.

## Data source and licence

Data: Olist, [Brazilian E-Commerce Public Dataset by Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)
and [Marketing Funnel by Olist](https://www.kaggle.com/datasets/olistbr/marketing-funnel-olist),
published on Kaggle under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).
Thanks to Olist for releasing it. The cleaned `analytics` tables are a derivative of this
data and are shared under the same licence.
