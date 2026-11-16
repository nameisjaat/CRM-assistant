"""Fixed rule text for the Generate SQL system prompt (step2-sql-lane-rules.md).

GLOBAL_RULES       Part C: traps in the data model; every query must follow them.
AMBIGUITY_DEFAULTS A3: what to assume when a question is vague (recorded as an assumption).
DATA_LIMITS        A4: what the data cannot answer (answerable = false).
SQL_RULES          B2: how the SQL itself must be written.

Plain text, sent to the LLM as is. Change a rule here and every prompt picks it up.
"""

GLOBAL_RULES = """\
1. Revenue = SUM(order_items.unit_price), excluding orders whose status is 'canceled' or \
'unavailable'. Freight is separate. Never compute revenue from order_payments.
2. Units sold = COUNT(*) of order_items. There is no quantity column: buying 3 of a product is 3 rows.
3. Number of orders = count from orders, never from order_items (some orders have no items).
4. Number of customers = COUNT(DISTINCT customer_id). customers has one row per real person.
5. Never join two of order_items, order_payments, order_reviews in the same SELECT. Aggregate \
each to one row per order_id in its own CTE first, then join the CTEs. Otherwise rows multiply \
and every sum and count is inflated.
6. Average review score uses order_reviews WHERE is_latest_for_order, so each order counts once.
7. Delivery questions (delivery time, late, on time) filter status = 'delivered' AND NOT has_date_issue.
8. Where an order went = orders.delivery_state / delivery_city. Where a customer lives = \
customers.state / city.
9. Area questions default to state. City names are lowercase without accents ('sao paulo').
10. Joins to geolocation are always LEFT JOIN. So is closed_deals -> sellers.
11. A lead converted if its mql_id appears in closed_deals. Conversion rate = deals / leads, \
using leads LEFT JOIN closed_deals.
12. Behaviour profiles can be combined ('cat, wolf'): match with ILIKE '%wolf%'.
13. Review text is in Portuguese. Keyword searches must use Portuguese words.
14. Money is BRL. Timestamps are Brazil local time.
15. Never output a product, person or page name. Only ids and categories exist."""

AMBIGUITY_DEFAULTS = """\
When the question is vague, use these defaults and list each one you used in `assumptions`:
- "Revenue", "sales": SUM(order_items.unit_price) over orders whose status is not 'canceled' \
or 'unavailable'; freight excluded.
- "This year", "last month", "recently": anchored to the latest orders.purchased_at in the \
data facts, not today's date. "This year" means the latest year in the data.
- No time period given: all data.
- "Top" with no number: top 10.
- "Area", "region", "where": state. City only if the question names a city.
- "Customers" in a count: real people, COUNT(DISTINCT customer_id).
- "Average rating", "satisfaction": average score over order_reviews WHERE is_latest_for_order.
- "Delivery time", "late": only status = 'delivered' AND NOT has_date_issue.
- A category in plain words ("health and beauty"): the matching listed value (health_beauty).
- "Sales rep": closed_deals.sr_id. "SDR" or "qualifier": closed_deals.sdr_id."""

DATA_LIMITS = """\
The data cannot answer questions that need any of these. For such a question, set \
answerable = false, give the reason in unanswerable_reason, and write no SQL. Never invent them:
- Product names or descriptions (removed by anonymization; only product_id and category exist).
- Names of customers, sellers or sales reps (anonymized ids only).
- Landing page names or URLs.
- Anything after the latest date in the data facts, or forecasts.
- Profit, cost or margin (only prices, freight and payments exist).
- Funnel data (leads, deals, conversion) outside the funnel period in the data facts."""

SQL_RULES = """\
1. One statement only, starting with SELECT or WITH. No semicolons.
2. Use only the tables listed in the schema below, by table name without a schema prefix. \
Never use raw, pg_catalog, information_schema or any other schema.
3. Name every column explicitly. No SELECT * (COUNT(*) is fine).
4. Alias every computed column with a readable snake_case name, ending as follows: money \
ends in revenue, price, amount or freight; counts end in _count or are named units; \
percentages end in _pct and run 0-100 (not 0-1); averages of review scores end in score; \
durations end in _days. A percentage change or growth that can be negative or above 100 \
ends in _change_pct or _growth_pct. A difference in days that can be negative (e.g. days \
late vs the estimate, negative when early) ends in _diff_days.
5. Round money and averages to 2 decimals in the output (round(x, 2)), but never round \
before aggregating.
6. Filter with the exact listed values (category = 'health_beauty', not 'Health & Beauty').
7. Add ORDER BY whenever the question implies an order ("top", "most", "by month"), and \
LIMIT for "top N".
8. expected_shape must match the SQL: single_value = one row and one column; list = rows of \
items; grouped = one row per group; time_series = one row per period, ordered by period. \
max_rows is the most rows the query can return (the LIMIT), or null when there is no limit. \
order_matters is true when the question implies an order.
9. Postgres syntax (e.g. date_trunc, extract, ::numeric casts)."""
