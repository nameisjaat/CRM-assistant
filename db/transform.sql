-- Step 1 transform: drop and rebuild the analytics schema from raw.
-- Source of truth: step1-data-model-olist.md. Run after load_raw.py and checks.sql:
--     python db/run_sql.py db/transform.sql
-- raw is only read, never changed. Safe to run again: analytics is dropped and rebuilt.
--
-- Changes from the spec's SQL, approved after the Phase 3 checks:
--   * sellers: the city is also split on a backslash ("rio de janeiro \rio de janeiro"), and a
--     city that is only a state code ("sp") or the abbreviation "sbc" is treated as invalid and
--     taken from geolocation, like cities with digits or @.
--   * closed_deals: deals won before the lead's first contact are left out (1 row in the data).
--   * leads: the table comment's date placeholder is filled from leads check 3.
--   * Comments added for every column the spec leaves without one.

-- ============================================================
-- Setup
-- ============================================================

\set ON_ERROR_STOP on
SET TIME ZONE 'America/Sao_Paulo';   -- Olist timestamps are Brazil local time with no zone

DROP SCHEMA IF EXISTS analytics CASCADE;
CREATE SCHEMA analytics;
CREATE EXTENSION IF NOT EXISTS unaccent;

-- ============================================================
-- 1. geolocation
-- ============================================================

CREATE TABLE analytics.geolocation AS
SELECT lpad(geolocation_zip_code_prefix::text, 5, '0')                       AS zip_prefix,
       mode() WITHIN GROUP (ORDER BY lower(unaccent(trim(geolocation_city)))) AS city,
       mode() WITHIN GROUP (ORDER BY upper(geolocation_state))               AS state
FROM raw.geolocation
GROUP BY 1;

ALTER TABLE analytics.geolocation ADD PRIMARY KEY (zip_prefix);

COMMENT ON TABLE analytics.geolocation IS
  'One row per 5-digit zip prefix: its city and state. Some customer/seller zips are missing here, so LEFT JOIN.';
COMMENT ON COLUMN analytics.geolocation.zip_prefix IS
  '5-digit zip prefix as text with leading zeros, e.g. 01037. Join from customers.zip_prefix, sellers.zip_prefix or orders.delivery_zip_prefix with a LEFT JOIN.';
COMMENT ON COLUMN analytics.geolocation.city IS
  'Most common city for this zip prefix, lowercase without accents, e.g. sao paulo.';
COMMENT ON COLUMN analytics.geolocation.state IS
  'Two-letter Brazilian state code for this zip prefix, e.g. SP, RJ.';

-- ============================================================
-- 2. orders
-- ============================================================

CREATE TABLE analytics.orders AS
WITH base AS (
    SELECT o.order_id,
           c.customer_unique_id                                     AS customer_id,
           o.order_status                                           AS status,
           o.order_purchase_timestamp::timestamptz                  AS purchased_at,
           NULLIF(o.order_approved_at, '')::timestamptz             AS approved_at,
           NULLIF(o.order_delivered_carrier_date, '')::timestamptz  AS shipped_at,
           NULLIF(o.order_delivered_customer_date, '')::timestamptz AS delivered_at,
           o.order_estimated_delivery_date::timestamptz             AS estimated_delivery_at,
           lpad(c.customer_zip_code_prefix::text, 5, '0')           AS delivery_zip_prefix,
           lower(unaccent(trim(c.customer_city)))                   AS delivery_city,
           upper(c.customer_state)                                  AS delivery_state
    FROM raw.orders o
    JOIN raw.customers c ON c.customer_id = o.customer_id
)
SELECT base.*,
       round((extract(epoch FROM delivered_at - purchased_at) / 86400)::numeric, 1) AS delivery_days,
       delivered_at::date > estimated_delivery_at::date                             AS is_late,
       coalesce((status = 'delivered' AND delivered_at IS NULL)
             OR delivered_at < purchased_at
             OR approved_at  < purchased_at, false)                                 AS has_date_issue
FROM base;
-- Derived columns are computed here, not as GENERATED columns: casting timestamptz to date
-- depends on the session time zone, and Postgres rejects that in a generated column.
-- has_date_issue is coalesced to false so that WHERE NOT has_date_issue keeps normal orders.

ALTER TABLE analytics.orders ADD PRIMARY KEY (order_id);

COMMENT ON TABLE analytics.orders IS
  'One row per order. For delivery questions, filter status = ''delivered'' and exclude has_date_issue. Date columns are NULL when the order has not reached that stage.';
COMMENT ON COLUMN analytics.orders.order_id IS
  'Unique order id. Count this for number of orders. Join: order_items / order_payments / order_reviews.order_id = orders.order_id.';
COMMENT ON COLUMN analytics.orders.customer_id IS
  'The real person who placed the order. Join: orders.customer_id = customers.customer_id.';
COMMENT ON COLUMN analytics.orders.status IS
  'Order status: delivered, shipped, canceled, unavailable, invoiced, processing, created, approved.';
COMMENT ON COLUMN analytics.orders.purchased_at IS
  'When the customer placed the order (Brazil local time). Never NULL. Use this for orders or revenue by day, month or year.';
COMMENT ON COLUMN analytics.orders.approved_at IS
  'When payment was approved. NULL if never approved (e.g. cancelled before payment).';
COMMENT ON COLUMN analytics.orders.shipped_at IS
  'When the order was handed to the carrier. NULL if not shipped yet or cancelled.';
COMMENT ON COLUMN analytics.orders.delivered_at IS
  'When the customer received the order. NULL if not delivered yet, cancelled, or lost.';
COMMENT ON COLUMN analytics.orders.estimated_delivery_at IS
  'Delivery date promised to the customer at purchase (date only, stored as midnight Brazil time). Never NULL. is_late compares delivered_at with this.';
COMMENT ON COLUMN analytics.orders.delivery_zip_prefix IS
  '5-digit zip prefix the order was delivered to, as text with leading zeros. LEFT JOIN to geolocation (some zips are missing there).';
COMMENT ON COLUMN analytics.orders.delivery_city IS
  'City the order was delivered to, lowercase without accents, e.g. sao paulo.';
COMMENT ON COLUMN analytics.orders.delivery_days IS
  'Days from purchase to delivery. NULL if not delivered.';
COMMENT ON COLUMN analytics.orders.is_late IS
  'True if delivered after the estimated delivery date. NULL if not delivered.';
COMMENT ON COLUMN analytics.orders.has_date_issue IS
  'True for impossible date combinations; exclude these from delivery-time analysis.';
COMMENT ON COLUMN analytics.orders.delivery_state IS
  'State the order was delivered to. Use this for orders-by-area questions; customers.state is the customer''s home.';

-- ============================================================
-- 3. customers (built from orders), then the foreign key orders -> customers
-- ============================================================

CREATE TABLE analytics.customers AS
WITH orders_by_person AS (
    SELECT customer_id,
           delivery_zip_prefix AS zip_prefix,
           delivery_city       AS city,
           delivery_state      AS state,
           purchased_at,
           row_number() OVER (PARTITION BY customer_id
                              ORDER BY purchased_at DESC) AS recency
    FROM analytics.orders
),
last_three AS (
    SELECT customer_id, zip_prefix, city, state,
           count(*)          AS times_used,
           max(purchased_at) AS last_used
    FROM orders_by_person
    WHERE recency <= 3
    GROUP BY customer_id, zip_prefix, city, state
)
SELECT DISTINCT ON (customer_id)
       customer_id, zip_prefix, city, state
FROM last_three
ORDER BY customer_id, times_used DESC, last_used DESC;   -- mode, then most recent

ALTER TABLE analytics.customers ADD PRIMARY KEY (customer_id);

ALTER TABLE analytics.orders
  ADD FOREIGN KEY (customer_id) REFERENCES analytics.customers (customer_id);

COMMENT ON TABLE analytics.customers IS
  'One row per real customer. Home address = most common address among their last 3 orders, ties broken by most recent. For where a specific order went, use orders.delivery_*.';
COMMENT ON COLUMN analytics.customers.customer_id IS
  'Unique person id (Olist customer_unique_id). Count this for number of customers.';
COMMENT ON COLUMN analytics.customers.zip_prefix IS
  '5-digit zip prefix of the home address, as text with leading zeros.';
COMMENT ON COLUMN analytics.customers.city IS
  'Home city, lowercase without accents, e.g. sao paulo.';
COMMENT ON COLUMN analytics.customers.state IS
  'Two-letter Brazilian state code of the home address, e.g. SP, RJ. Preferred field for area questions.';

-- ============================================================
-- 4. products
-- ============================================================

CREATE TABLE analytics.products AS
SELECT p.product_id,
       COALESCE(t.product_category_name_english,
                NULLIF(p.product_category_name, ''),
                'unknown')                                         AS category,
       NULLIF(p.product_name_lenght, '')::numeric::int             AS name_length,
       NULLIF(p.product_description_lenght, '')::numeric::int      AS description_length,
       NULLIF(p.product_photos_qty, '')::numeric::int              AS photo_count,
       NULLIF(p.product_weight_g, '')::numeric::int                AS weight_g,
       NULLIF(p.product_length_cm, '')::numeric::int               AS length_cm,
       NULLIF(p.product_height_cm, '')::numeric::int               AS height_cm,
       NULLIF(p.product_width_cm, '')::numeric::int                AS width_cm
FROM raw.products p
LEFT JOIN raw.product_category_name_translation t
       ON t.product_category_name = p.product_category_name;

ALTER TABLE analytics.products
  ADD PRIMARY KEY (product_id),
  ADD COLUMN volume_cm3 int
      GENERATED ALWAYS AS (length_cm * height_cm * width_cm) STORED;

COMMENT ON TABLE analytics.products IS
  'One row per product. Product NAMES ARE NOT AVAILABLE (removed by anonymization): identify products by product_id and category only, never invent names.';
COMMENT ON COLUMN analytics.products.product_id IS
  'Unique product id (anonymized). The only way to identify a product; there are no product names.';
COMMENT ON COLUMN analytics.products.category IS
  'Product category in English, snake_case, e.g. health_beauty, bed_bath_table. ''unknown'' if missing. A few untranslated categories keep their Portuguese name.';
COMMENT ON COLUMN analytics.products.name_length IS
  'Number of characters in the product name. The name itself is not available.';
COMMENT ON COLUMN analytics.products.description_length IS
  'Number of characters in the product description. The description itself is not available.';
COMMENT ON COLUMN analytics.products.photo_count IS
  'Number of photos on the product listing.';
COMMENT ON COLUMN analytics.products.weight_g IS
  'Package weight in grams. NULL if missing.';
COMMENT ON COLUMN analytics.products.length_cm IS
  'Package length in centimetres. NULL if missing.';
COMMENT ON COLUMN analytics.products.height_cm IS
  'Package height in centimetres. NULL if missing.';
COMMENT ON COLUMN analytics.products.width_cm IS
  'Package width in centimetres. NULL if missing.';
COMMENT ON COLUMN analytics.products.volume_cm3 IS
  'Package volume in cubic centimetres (length x height x width). NULL if any dimension is missing.';

-- ============================================================
-- 5. sellers
-- ============================================================

CREATE TABLE analytics.sellers AS
WITH cleaned AS (
    SELECT seller_id,
           lpad(seller_zip_code_prefix::text, 5, '0') AS zip_prefix,
           seller_city                                AS raw_city,
           lower(unaccent(trim(
               split_part(split_part(split_part(seller_city, '/', 1), ',', 1), '\', 1)
           )))                                        AS city,
           upper(trim(seller_state))                  AS state
    FROM raw.sellers
)
SELECT c.seller_id,
       c.zip_prefix,
       CASE WHEN c.raw_city ~ '[0-9@]'                -- not a real city: use geolocation
              OR upper(c.city) IN ('AC','AL','AP','AM','BA','CE','DF','ES','GO','MA','MT','MS','MG','PA',
                                   'PB','PR','PE','PI','RJ','RN','RS','RO','RR','SC','SP','SE','TO')
              OR c.city = 'sbc'                       -- state code alone, or an abbreviation
            THEN g.city
            ELSE c.city
       END     AS city,
       c.state
FROM cleaned c
LEFT JOIN analytics.geolocation g ON g.zip_prefix = c.zip_prefix;

ALTER TABLE analytics.sellers ADD PRIMARY KEY (seller_id);

COMMENT ON TABLE analytics.sellers IS
  'One row per seller on the Olist marketplace. Join to order_items for sales. Sellers acquired through the marketing funnel appear in closed_deals.seller_id (LEFT JOIN: not every deal has a seller here).';
COMMENT ON COLUMN analytics.sellers.seller_id IS
  'Unique seller id (anonymized; no seller names available). Join: order_items.seller_id = sellers.seller_id.';
COMMENT ON COLUMN analytics.sellers.zip_prefix IS
  '5-digit zip prefix as text with leading zeros. LEFT JOIN to geolocation (some zips are missing there).';
COMMENT ON COLUMN analytics.sellers.city IS
  'Seller city, lowercase without accents, e.g. sao paulo. State suffixes removed; invalid values replaced from geolocation.';
COMMENT ON COLUMN analytics.sellers.state IS
  'Two-letter Brazilian state code, e.g. SP. Preferred field for seller-area questions.';

-- ============================================================
-- 6. order_items
-- ============================================================

CREATE TABLE analytics.order_items AS
SELECT order_id,
       order_item_id::int                 AS item_number,
       product_id,
       seller_id,
       shipping_limit_date::timestamptz   AS ship_by_at,
       price::numeric(10,2)               AS unit_price,
       freight_value::numeric(10,2)       AS freight
FROM raw.order_items;

ALTER TABLE analytics.order_items
  ADD PRIMARY KEY (order_id, item_number),
  ADD COLUMN item_total numeric(10,2) GENERATED ALWAYS AS (unit_price + freight) STORED,
  ADD FOREIGN KEY (order_id)   REFERENCES analytics.orders   (order_id),
  ADD FOREIGN KEY (product_id) REFERENCES analytics.products (product_id),
  ADD FOREIGN KEY (seller_id)  REFERENCES analytics.sellers  (seller_id);

COMMENT ON TABLE analytics.order_items IS
  'One row per UNIT sold. There is no quantity column: buying 3 of a product = 3 rows. Units = COUNT(*). Revenue = SUM(unit_price), excluding orders with status canceled or unavailable. Count orders from the orders table, not here (some orders have no items).';
COMMENT ON COLUMN analytics.order_items.order_id IS
  'Order this unit belongs to. Join: order_items.order_id = orders.order_id.';
COMMENT ON COLUMN analytics.order_items.item_number IS
  'Position of this unit within the order (1, 2, 3...). Not a unique id; the key is (order_id, item_number).';
COMMENT ON COLUMN analytics.order_items.product_id IS
  'Product sold. Join: order_items.product_id = products.product_id (for category).';
COMMENT ON COLUMN analytics.order_items.seller_id IS
  'Seller of this unit. One order can contain items from several sellers.';
COMMENT ON COLUMN analytics.order_items.unit_price IS
  'Price of one unit in BRL, excluding shipping.';
COMMENT ON COLUMN analytics.order_items.freight IS
  'Shipping cost for this unit in BRL. 0 means free shipping.';
COMMENT ON COLUMN analytics.order_items.item_total IS
  'unit_price + freight: what the customer paid for this unit including shipping.';
COMMENT ON COLUMN analytics.order_items.ship_by_at IS
  'Deadline for the seller to hand this item to the carrier.';

-- ============================================================
-- 7. order_payments
-- ============================================================

CREATE TABLE analytics.order_payments AS
SELECT order_id,
       payment_sequential::int                             AS payment_number,
       CASE WHEN payment_type = 'not_defined' THEN 'unknown'
            ELSE payment_type END                          AS method,
       NULLIF(payment_installments::numeric::int, 0)       AS installments,
       payment_value::numeric(10,2)                        AS amount
FROM raw.order_payments;

ALTER TABLE analytics.order_payments
  ADD PRIMARY KEY (order_id, payment_number),
  ADD FOREIGN KEY (order_id) REFERENCES analytics.orders (order_id);

COMMENT ON TABLE analytics.order_payments IS
  'One row per payment; an order can have several (e.g. voucher + credit card). Use for HOW customers pay. Revenue comes from order_items, not here. NEVER join order_payments and order_items in the same query without first aggregating each to one row per order_id, or totals are multiplied.';
COMMENT ON COLUMN analytics.order_payments.order_id IS
  'Order this payment is for. Join: order_payments.order_id = orders.order_id.';
COMMENT ON COLUMN analytics.order_payments.payment_number IS
  'Position of this payment within the order (1, 2, 3...). Key is (order_id, payment_number).';
COMMENT ON COLUMN analytics.order_payments.method IS
  'Payment method: credit_card, boleto (Brazilian bank slip), voucher, debit_card, unknown.';
COMMENT ON COLUMN analytics.order_payments.installments IS
  'Number of instalments the payment was split into (e.g. 10 monthly payments). Not an amount. NULL if invalid.';
COMMENT ON COLUMN analytics.order_payments.amount IS
  'Total amount of this payment in BRL (not per instalment). Sum per order for total paid.';

-- ============================================================
-- 8. order_reviews
-- ============================================================

CREATE TABLE analytics.order_reviews AS
SELECT review_id,
       order_id,
       review_score::int                              AS score,
       NULLIF(trim(review_comment_title), '')         AS title,
       NULLIF(trim(review_comment_message), '')       AS message,
       review_creation_date::timestamptz              AS sent_at,
       review_answer_timestamp::timestamptz           AS answered_at,
       row_number() OVER (PARTITION BY order_id
                          ORDER BY review_answer_timestamp::timestamptz DESC, review_id) = 1
                                                      AS is_latest_for_order
FROM (SELECT DISTINCT * FROM raw.order_reviews) r;    -- drops exact duplicate rows only

ALTER TABLE analytics.order_reviews
  ADD PRIMARY KEY (review_id, order_id),
  ADD FOREIGN KEY (order_id) REFERENCES analytics.orders (order_id),
  ADD CONSTRAINT score_range CHECK (score BETWEEN 1 AND 5),
  ADD COLUMN has_comment boolean
      GENERATED ALWAYS AS (title IS NOT NULL OR message IS NOT NULL) STORED;

COMMENT ON TABLE analytics.order_reviews IS
  'Customer satisfaction reviews. One review can cover several orders and one order can have several reviews; key is (review_id, order_id). For average score, use WHERE is_latest_for_order so each order counts once. NEVER join with order_items or order_payments without aggregating each to one row per order_id first.';
COMMENT ON COLUMN analytics.order_reviews.review_id IS
  'Review (survey answer) id. Not unique on its own: one review can cover several orders. Key is (review_id, order_id).';
COMMENT ON COLUMN analytics.order_reviews.order_id IS
  'Order reviewed. Join: order_reviews.order_id = orders.order_id. An order can have several reviews; filter is_latest_for_order to count each order once.';
COMMENT ON COLUMN analytics.order_reviews.score IS
  'Satisfaction score from 1 (worst) to 5 (best).';
COMMENT ON COLUMN analytics.order_reviews.is_latest_for_order IS
  'True for the most recently answered review of each order. Filter on this for per-order averages and counts.';
COMMENT ON COLUMN analytics.order_reviews.title IS
  'Optional review title, in PORTUGUESE. NULL if empty.';
COMMENT ON COLUMN analytics.order_reviews.message IS
  'Optional review comment, in PORTUGUESE. NULL if empty. Keyword searches must use Portuguese words.';
COMMENT ON COLUMN analytics.order_reviews.sent_at IS
  'When the satisfaction survey was sent to the customer.';
COMMENT ON COLUMN analytics.order_reviews.answered_at IS
  'When the customer answered the survey.';
COMMENT ON COLUMN analytics.order_reviews.has_comment IS
  'True if the customer wrote a title or a comment.';

-- ============================================================
-- 9. leads
-- ============================================================

CREATE TABLE analytics.leads AS
SELECT mql_id,
       first_contact_date::date                      AS first_contact_date,
       landing_page_id,
       coalesce(NULLIF(trim(origin), ''), 'unknown') AS channel
FROM raw.leads;

ALTER TABLE analytics.leads ADD PRIMARY KEY (mql_id);

COMMENT ON TABLE analytics.leads IS
  'Marketing qualified leads (MQLs): businesses that asked to become sellers on Olist. One row per lead. A lead CONVERTED if its mql_id appears in closed_deals; conversion rate = closed deals / leads. Covers 2017-06-14 to 2018-05-31 only.';
COMMENT ON COLUMN analytics.leads.mql_id IS
  'Unique lead id (MQL = marketing qualified lead, anonymized). Join: closed_deals.mql_id = leads.mql_id.';
COMMENT ON COLUMN analytics.leads.channel IS
  'Marketing channel the lead came from: organic_search, paid_search, social, direct_traffic, email, referral, display, other, other_publicities, unknown.';
COMMENT ON COLUMN analytics.leads.first_contact_date IS
  'Date the lead first contacted Olist (date only, no time).';
COMMENT ON COLUMN analytics.leads.landing_page_id IS
  'Anonymized id of the signup landing page. No page names or URLs are available.';

-- ============================================================
-- 10. closed_deals
-- ============================================================

CREATE TABLE analytics.closed_deals AS
SELECT d.mql_id,
       d.seller_id,
       NULLIF(trim(d.sdr_id), '')                                 AS sdr_id,
       NULLIF(trim(d.sr_id), '')                                  AS sr_id,
       d.won_date::timestamptz                                    AS won_at,
       d.won_date::date - l.first_contact_date::date              AS days_to_close,
       NULLIF(trim(d.business_segment), '')                       AS business_segment,
       NULLIF(trim(d.lead_type), '')                              AS lead_type,
       NULLIF(trim(d.lead_behaviour_profile), '')                 AS behaviour_profile,
       NULLIF(trim(d.business_type), '')                          AS business_type,
       NULLIF(NULLIF(d.declared_monthly_revenue, '')::numeric, 0) AS declared_monthly_revenue
FROM raw.closed_deals d
LEFT JOIN raw.leads l USING (mql_id)
WHERE l.first_contact_date IS NULL
   OR d.won_date::date >= l.first_contact_date::date;   -- approved: drop deals won before first contact (data error)
-- dropped (mostly empty, kept in raw): has_company, has_gtin, average_stock, declared_product_catalog_size

ALTER TABLE analytics.closed_deals
  ADD PRIMARY KEY (mql_id),
  ADD FOREIGN KEY (mql_id) REFERENCES analytics.leads (mql_id);

COMMENT ON TABLE analytics.closed_deals IS
  'Leads that became sellers (won deals). One row per deal, keyed by mql_id (joins to leads). seller_id joins to sellers with a LEFT JOIN: many deal sellers never sold anything, so revenue from funnel sellers covers only those with orders. Deals recorded as won before the lead''s first contact are excluded as data errors.';
COMMENT ON COLUMN analytics.closed_deals.mql_id IS
  'Lead that became this deal. Primary key; join: closed_deals.mql_id = leads.mql_id.';
COMMENT ON COLUMN analytics.closed_deals.seller_id IS
  'Seller account created from the deal. LEFT JOIN to sellers: many deal sellers never sold anything, so they are not in sellers.';
COMMENT ON COLUMN analytics.closed_deals.sdr_id IS
  'Sales development rep who qualified the lead. Anonymized id; no names available.';
COMMENT ON COLUMN analytics.closed_deals.sr_id IS
  'Sales rep who closed the deal. Anonymized id; no names available.';
COMMENT ON COLUMN analytics.closed_deals.won_at IS
  'When the deal was won.';
COMMENT ON COLUMN analytics.closed_deals.days_to_close IS
  'Days from the lead''s first contact to the deal being won.';
COMMENT ON COLUMN analytics.closed_deals.business_segment IS
  'Seller''s market segment, e.g. home_decor, health_beauty. Different from products.category.';
COMMENT ON COLUMN analytics.closed_deals.lead_type IS
  'Kind of business, e.g. online_medium, online_big, industry, offline.';
COMMENT ON COLUMN analytics.closed_deals.behaviour_profile IS
  'Personality profile (cat, eagle, wolf, shark) or a combination like ''cat, wolf''. Use ILIKE ''%wolf%'' to include combinations.';
COMMENT ON COLUMN analytics.closed_deals.business_type IS
  'Seller''s business type: reseller, manufacturer, other. NULL if not recorded.';
COMMENT ON COLUMN analytics.closed_deals.declared_monthly_revenue IS
  'Monthly revenue in BRL as self-reported by the seller. NULL if not declared (most deals).';

-- ============================================================
-- Grants
-- ============================================================

DO $$ BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'crm_readonly') THEN
    CREATE ROLE crm_readonly LOGIN;
  END IF;
END $$;
-- password is set by the runner from READONLY_PASSWORD: ALTER ROLE crm_readonly PASSWORD '...';

REVOKE ALL ON SCHEMA raw FROM crm_readonly;
GRANT USAGE ON SCHEMA analytics TO crm_readonly;
GRANT SELECT ON ALL TABLES IN SCHEMA analytics TO crm_readonly;

ALTER ROLE crm_readonly SET search_path = analytics;
ALTER ROLE crm_readonly SET default_transaction_read_only = on;
ALTER ROLE crm_readonly SET statement_timeout = '5s';
ALTER ROLE crm_readonly SET timezone = 'America/Sao_Paulo';
