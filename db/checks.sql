-- Step 1 checks: confirm the assumptions behind the decisions in step1-data-model-olist.md.
-- Read-only queries against raw. Run after loading raw and before transform.sql:
--     python db/run_sql.py db/checks.sql
-- If a result contradicts its "expect" note, stop and report it instead of changing a decision.

-- ============================================================
-- Geolocation
-- ============================================================

\echo 'Geolocation check 1: zips mapping to more than one city or state (a handful is normal)'
SELECT geolocation_zip_code_prefix,
       count(DISTINCT lower(unaccent(trim(geolocation_city)))) AS cities,
       count(DISTINCT geolocation_state)                      AS states
FROM raw.geolocation
GROUP BY 1
HAVING count(DISTINCT geolocation_state) > 1
    OR count(DISTINCT lower(unaccent(trim(geolocation_city)))) > 1;

-- ============================================================
-- Customers
-- ============================================================

\echo 'Customers check 1: rows vs real people (expect: rows = per-order ids > people)'
SELECT count(*)                            AS rows,
       count(DISTINCT customer_id)         AS per_order_ids,
       count(DISTINCT customer_unique_id)  AS people
FROM raw.customers;

\echo 'Customers check 2: how many people ordered more than once'
SELECT count(*) FROM (
  SELECT customer_unique_id FROM raw.customers
  GROUP BY 1 HAVING count(*) > 1) t;

\echo 'Customers check 3: how many people have more than one address across orders'
SELECT count(*) FROM (
  SELECT customer_unique_id FROM raw.customers
  GROUP BY 1 HAVING count(DISTINCT customer_zip_code_prefix) > 1) t;

\echo 'Customers check 4: customer zips missing from geolocation'
SELECT count(DISTINCT customer_zip_code_prefix) FROM raw.customers c
WHERE NOT EXISTS (SELECT 1 FROM raw.geolocation g
                  WHERE g.geolocation_zip_code_prefix = c.customer_zip_code_prefix);

-- ============================================================
-- Orders
-- ============================================================

\echo 'Orders check 1: confirm the 1:1 (1:1 if both numbers match the row count of raw.customers)'
SELECT count(*)                     AS orders,
       count(DISTINCT customer_id)  AS per_order_customer_ids
FROM raw.orders;

\echo 'Orders check 2: where the date gaps fall by status (expect only a small handful in delivered)'
SELECT order_status,
       count(*)                                                       AS orders,
       count(*) FILTER (WHERE order_approved_at IS NULL)              AS no_approval,
       count(*) FILTER (WHERE order_delivered_carrier_date IS NULL)   AS no_carrier,
       count(*) FILTER (WHERE order_delivered_customer_date IS NULL)  AS no_delivery
FROM raw.orders
GROUP BY order_status
ORDER BY orders DESC;

-- ============================================================
-- Products
-- ============================================================

\echo 'Products check 1: product_id is unique (expect 0)'
SELECT count(*) - count(DISTINCT product_id) FROM raw.products;

\echo 'Products check 2: missing values per column'
SELECT count(*) FILTER (WHERE product_category_name IS NULL OR product_category_name = '') AS no_category,
       count(*) FILTER (WHERE product_photos_qty    IS NULL OR product_photos_qty    = '') AS no_photos,
       count(*) FILTER (WHERE product_weight_g      IS NULL OR product_weight_g      = '') AS no_weight,
       count(*) FILTER (WHERE product_length_cm     IS NULL OR product_length_cm     = '') AS no_dimensions
FROM raw.products;

\echo 'Products check 3: categories with no English translation'
SELECT DISTINCT p.product_category_name
FROM raw.products p
LEFT JOIN raw.product_category_name_translation t
       ON t.product_category_name = p.product_category_name
WHERE p.product_category_name <> '' AND t.product_category_name IS NULL;

\echo 'Products check 4: impossible sizes (zero or negative weight / dimensions)'
SELECT count(*) FROM raw.products
WHERE product_weight_g::numeric <= 0 OR product_length_cm::numeric <= 0
   OR product_height_cm::numeric <= 0 OR product_width_cm::numeric <= 0;

-- ============================================================
-- Sellers
-- ============================================================

\echo 'Sellers check 1: seller_id is unique (expect 0)'
SELECT count(*) - count(DISTINCT seller_id) FROM raw.sellers;

\echo 'Sellers check 2: suspicious city values: digits, @, slashes, commas'
SELECT seller_city, seller_state, count(*)
FROM raw.sellers
WHERE seller_city ~ '[0-9@/,\\]'
GROUP BY 1, 2 ORDER BY 3 DESC;

\echo 'Sellers check 3: seller zips missing from geolocation'
SELECT count(*) FROM raw.sellers s
WHERE NOT EXISTS (SELECT 1 FROM raw.geolocation g
                  WHERE g.geolocation_zip_code_prefix = s.seller_zip_code_prefix);

\echo 'Sellers check 4: sellers with no items sold (expect 0 or close to it)'
SELECT count(*) FROM raw.sellers s
WHERE NOT EXISTS (SELECT 1 FROM raw.order_items i WHERE i.seller_id = s.seller_id);

\echo 'Sellers check 5: how many sellers came through the marketing funnel'
SELECT count(*) FROM raw.sellers s
WHERE EXISTS (SELECT 1 FROM raw.closed_deals d WHERE d.seller_id = s.seller_id);

-- ============================================================
-- Order items
-- ============================================================

\echo 'Order items check 1: composite key is unique (expect 0)'
SELECT count(*) FROM (
  SELECT order_id, order_item_id FROM raw.order_items
  GROUP BY 1, 2 HAVING count(*) > 1) t;

\echo 'Order items check 2: orphans: items pointing to missing orders / products / sellers (expect 0 each)'
SELECT
  (SELECT count(*) FROM raw.order_items i WHERE NOT EXISTS (SELECT 1 FROM raw.orders   o WHERE o.order_id   = i.order_id))   AS no_order,
  (SELECT count(*) FROM raw.order_items i WHERE NOT EXISTS (SELECT 1 FROM raw.products p WHERE p.product_id = i.product_id)) AS no_product,
  (SELECT count(*) FROM raw.order_items i WHERE NOT EXISTS (SELECT 1 FROM raw.sellers  s WHERE s.seller_id  = i.seller_id))  AS no_seller;

\echo 'Order items check 3: odd values (expect 0 bad values; zero freight is normal = free shipping)'
SELECT count(*) FILTER (WHERE price::numeric <= 0)         AS bad_price,
       count(*) FILTER (WHERE freight_value::numeric < 0)  AS bad_freight,
       count(*) FILTER (WHERE freight_value::numeric = 0)  AS free_shipping
FROM raw.order_items;

\echo 'Order items check 4: orders with no items at all (likely cancelled / unavailable)'
SELECT o.order_status, count(*)
FROM raw.orders o
WHERE NOT EXISTS (SELECT 1 FROM raw.order_items i WHERE i.order_id = o.order_id)
GROUP BY 1;

-- ============================================================
-- Order payments
-- ============================================================

\echo 'Order payments check 1: composite key is unique (expect 0)'
SELECT count(*) FROM (
  SELECT order_id, payment_sequential FROM raw.order_payments
  GROUP BY 1, 2 HAVING count(*) > 1) t;

\echo 'Order payments check 2: payment types, and how many have zero value'
SELECT payment_type,
       count(*)                                              AS payments,
       count(*) FILTER (WHERE payment_value::numeric = 0)    AS zero_value
FROM raw.order_payments
GROUP BY 1 ORDER BY 2 DESC;

\echo 'Order payments check 3: odd instalment counts (0 is impossible)'
SELECT payment_type, payment_installments::int AS installments, count(*)
FROM raw.order_payments
WHERE payment_installments::int = 0 OR payment_installments::int > 24
GROUP BY 1, 2;

\echo 'Order payments check 4: orders with no payment, and payments for missing orders (expect small / 0)'
SELECT
  (SELECT count(*) FROM raw.orders o
   WHERE NOT EXISTS (SELECT 1 FROM raw.order_payments p WHERE p.order_id = o.order_id)) AS orders_without_payment,
  (SELECT count(*) FROM raw.order_payments p
   WHERE NOT EXISTS (SELECT 1 FROM raw.orders o WHERE o.order_id = p.order_id))         AS payments_without_order;

\echo 'Order payments check 5: do payments match what was charged? (each side summed per order first)'
WITH paid  AS (SELECT order_id, sum(payment_value::numeric) AS paid
               FROM raw.order_payments GROUP BY 1),
     items AS (SELECT order_id, sum(price::numeric + freight_value::numeric) AS charged
               FROM raw.order_items GROUP BY 1)
SELECT count(*)                                             AS orders_compared,
       count(*) FILTER (WHERE abs(paid - charged) > 1)      AS differ_by_over_1_brl
FROM paid JOIN items USING (order_id);

-- ============================================================
-- Order reviews
-- ============================================================

\echo 'Order reviews check 1: what is unique? (expect: pairs = rows, review_ids and order_ids < rows)'
SELECT count(*)                                      AS rows,
       count(DISTINCT review_id)                     AS review_ids,
       count(DISTINCT order_id)                      AS order_ids,
       count(DISTINCT (review_id, order_id))         AS pairs
FROM raw.order_reviews;

\echo 'Order reviews check 2: scores outside 1-5 (expect 0)'
SELECT count(*) FROM raw.order_reviews
WHERE review_score::int NOT BETWEEN 1 AND 5;

\echo 'Order reviews check 3: how often is there any written comment?'
SELECT count(*) FILTER (WHERE coalesce(trim(review_comment_title), '')   = '') AS no_title,
       count(*) FILTER (WHERE coalesce(trim(review_comment_message), '') = '') AS no_message,
       count(*)                                                               AS total
FROM raw.order_reviews;

\echo 'Order reviews check 4: reviews for missing orders (expect 0) and orders with no review'
SELECT
  (SELECT count(*) FROM raw.order_reviews r
   WHERE NOT EXISTS (SELECT 1 FROM raw.orders o WHERE o.order_id = r.order_id)) AS reviews_without_order,
  (SELECT count(*) FROM raw.orders o
   WHERE NOT EXISTS (SELECT 1 FROM raw.order_reviews r WHERE r.order_id = o.order_id)) AS orders_without_review;

\echo 'Order reviews check 5: answered before the survey was sent (expect 0)'
SELECT count(*) FROM raw.order_reviews
WHERE review_answer_timestamp::timestamptz < review_creation_date::timestamptz;

-- ============================================================
-- Leads
-- ============================================================

\echo 'Leads check 1: mql_id is unique (expect 0)'
SELECT count(*) - count(DISTINCT mql_id) FROM raw.leads;

\echo 'Leads check 2: channel breakdown, including empty values'
SELECT coalesce(NULLIF(origin, ''), '<empty>') AS origin, count(*)
FROM raw.leads GROUP BY 1 ORDER BY 2 DESC;

\echo 'Leads check 3: date range of the funnel'
SELECT min(first_contact_date::date), max(first_contact_date::date) FROM raw.leads;

\echo 'Leads check 4: landing pages: how many, and leads per page'
SELECT count(DISTINCT landing_page_id) AS pages,
       round(count(*)::numeric / count(DISTINCT landing_page_id), 1) AS leads_per_page
FROM raw.leads;

\echo 'Leads check 5: deals that point to a lead that does not exist (expect 0)'
SELECT count(*) FROM raw.closed_deals d
WHERE NOT EXISTS (SELECT 1 FROM raw.leads l WHERE l.mql_id = d.mql_id);

-- ============================================================
-- Closed deals
-- ============================================================

\echo 'Closed deals check 1: one deal per lead (expect 0)'
SELECT count(*) - count(DISTINCT mql_id) FROM raw.closed_deals;

\echo 'Closed deals check 2: how empty is each column? (decides what to keep)'
SELECT count(*) AS deals,
  count(*) FILTER (WHERE coalesce(trim(business_segment), '')        = '') AS no_segment,
  count(*) FILTER (WHERE coalesce(trim(lead_type), '')               = '') AS no_lead_type,
  count(*) FILTER (WHERE coalesce(trim(lead_behaviour_profile), '')  = '') AS no_profile,
  count(*) FILTER (WHERE coalesce(trim(business_type), '')           = '') AS no_business_type,
  count(*) FILTER (WHERE coalesce(trim(has_company), '')             = '') AS no_has_company,
  count(*) FILTER (WHERE coalesce(trim(has_gtin), '')                = '') AS no_has_gtin,
  count(*) FILTER (WHERE coalesce(trim(average_stock), '')           = '') AS no_avg_stock,
  count(*) FILTER (WHERE coalesce(trim(declared_product_catalog_size), '') = '') AS no_catalog_size,
  count(*) FILTER (WHERE coalesce(NULLIF(declared_monthly_revenue, '')::numeric, 0) = 0) AS no_revenue
FROM raw.closed_deals;

\echo 'Closed deals check 3: how many deal sellers actually sold anything'
SELECT count(*) AS deals,
       count(*) FILTER (WHERE EXISTS (SELECT 1 FROM raw.sellers s
                                      WHERE s.seller_id = d.seller_id)) AS seller_has_orders
FROM raw.closed_deals d;

\echo 'Closed deals check 4: won before first contact (expect 0)'
SELECT count(*) FROM raw.closed_deals d
JOIN raw.leads l USING (mql_id)
WHERE d.won_date::date < l.first_contact_date::date;

\echo 'Closed deals check 5: values of the category columns'
SELECT 'segment' AS col, business_segment AS value, count(*) FROM raw.closed_deals GROUP BY 1, 2
UNION ALL
SELECT 'lead_type', lead_type, count(*) FROM raw.closed_deals GROUP BY 1, 2
UNION ALL
SELECT 'profile', lead_behaviour_profile, count(*) FROM raw.closed_deals GROUP BY 1, 2
ORDER BY 1, 3 DESC;
