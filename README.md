# CRM Analytics Assistant

A chat assistant that lets non-technical people ask plain-English questions about sales and
CRM data and get accurate answers, grounded only in the results of real, read-only SQL
queries.

> "Which 5 product categories had the most revenue in 2018?"
> "What about in 2017?"
> "How did revenue, number of orders and average review score change from 2017 to 2018?"

Built with LangGraph, OpenAI models, sqlglot and Postgres (Neon), on Olist's public
Brazilian e-commerce and marketing funnel data.

---

## 1. Problem statement

Business users have questions about their data ("which sellers grew the most?", "is the
marketing funnel converting?") but usually can't write SQL, and analysts become a queue.
LLMs can write SQL, but used naively they are dangerous and unreliable:

- **They can run harmful queries.** A generated `DELETE`, `DROP` or `pg_sleep` must never reach
  the database.
- **They make silent mistakes.** The SQL runs fine but the number is wrong: revenue that
  includes canceled orders, averages that count an order once per review, sums inflated by
  joining two one-to-many tables, `'cancelled'` instead of the stored `'canceled'`.
- **They invent things.** Product names, profit margins or forecasts that the data doesn't
  contain, and numbers in the final answer that no query returned.
- **Real questions aren't one query.** "Why did review scores drop?" or "top sellers, and
  also the most common payment method" need several queries and a combined answer.
- **Conversations have follow-ups.** "What about 2017?" only makes sense with memory.

The goal: an assistant that answers correctly when it can, says clearly when it can't, never
writes to the database, and never states a number that didn't come from a query.

## 2. Solution

A LangGraph chatbot with a **router** on top of two **lanes** and one shared **synthesizer**:

- **Router:** decides whether the question needs one query (`sql_query`), several queries
  (`analysis`), or isn't about the data (`out_of_scope`), and rewrites follow-ups into
  standalone questions using the conversation history.
- **sql_query lane:** turns one question into one validated, read-only SQL result through a
  self-correcting loop: schema selection, SQL generation, a two-layer review (deterministic
  code checks, then an LLM reviewer), read-only execution and result validation, with up to
  3 attempts that feed every error back to the generator.
- **analysis lane:** a planner splits a complex question into up to 6 independent
  sub-questions and runs each through the sql_query lane **in parallel**.
- **Synthesizer:** writes one answer from the results of either lane, and a code check
  verifies that **every number in the answer appears in a query result** (or in an earlier
  grounded answer). If not, the answer is rewritten once, then flagged.
- **Short-term memory:** the conversation is kept by a LangGraph checkpointer; the router,
  planner and synthesizer see the last 10 messages.

Safety is layered: the code review rejects anything that isn't a single read-only `SELECT`,
the database session is a read-only transaction with a 5-second timeout that is always rolled
back, and every query is capped at 1,000 rows.

## 3. Architecture

### 3.1 Chatbot (top level)

```
                         ┌ sql_query ──── run_sql ─────────────┐
question → start → router┤                                     ├→ synthesize ⇄ (1 retry) → answer
                         ├ analysis ───── run_analysis ────────┘     (grounding check)
                         └ out_of_scope ─ fixed reply ───────────────────────────────→ answer
```

| Node | Type | What it does |
| --- | --- | --- |
| `start` | code | Appends the question to `messages`, clears per-turn fields |
| `router` | LLM (gpt-4o-mini) | Route + standalone question, from the question and the last 10 messages |
| `run_sql` | lane | The sql_query lane on the standalone question |
| `run_analysis` | lane | The analysis lane, with the conversation history |
| `out_of_scope` | code | A fixed reply, no LLM |
| `synthesize` | LLM (gpt-4o-mini) + code | One answer for both lanes; grounding check with one retry |

### 3.2 sql_query lane

```
question → schema select → generate SQL → review → execute → validate → result
                               ↑            │         │          │
                               └─── error fed back (max 3 attempts) ┘
```

| Step | Type | What it does |
| --- | --- | --- |
| **Schema select** | LLM (gpt-4o-mini) + code | The LLM picks tables; code adds join tables (never through `geolocation`), adds `orders` for its child tables, caps at 6 tables, and builds the schema context (columns, descriptions, allowed values, joins) from a schema catalog generated from the database |
| **Generate SQL** | LLM (gpt-4o) | Structured output: `answerable`, `assumptions`, `sql`, `expected_shape`. The prompt carries 15 data-model rules, ambiguity defaults ("top" = 10, "this year" = latest year in the data), data limits and SQL-writing rules. Retries see every earlier SQL with its error |
| **Review, layer 1** | code (sqlglot) | 15 checks on the parsed SQL. Safety: one statement, read-only, known tables and columns, no dangerous functions, no placeholders. Data model: no row multiplication, no `SELECT *`, LEFT JOIN to geolocation, `ILIKE` for profiles, revenue excludes canceled orders and freight, average score uses the latest review, delivery filters, rounding after aggregation, percentages on 0-100 |
| **Review, layer 2** | LLM (gpt-4o-mini) + code | Checks only what depends on the question (units, order and customer counts, customer vs delivery location, state vs city, conversion, Portuguese keywords, exact filter values, ORDER BY / LIMIT N, does it answer the question). Every quote it gives is verified against the SQL by code, so invented problems are ignored |
| **Execute** | database | Shared connection; per run: read-only transaction, 5 s timeout, `search_path = analytics`, Brazil time zone, rollback. Fetches at most 1,001 rows, converts values to JSON-safe types. SQL errors and timeouts are fed back; connection failures are raised |
| **Validate** | code | Shape vs `expected_shape`, row cap, empty result (retried once, then accepted as "no matching data"), all-NULL columns, ORDER BY when order matters, impossible values by column name (negative revenue, score outside 1-5, percentage outside 0-100) |

### 3.3 analysis lane

```
question + history → planner ⇄ (1 replan after code check) → run_task × N (parallel, Send) → tasks + results
```

- **Planner** (gpt-4o-mini): 1-6 independent, standalone sub-questions. Dependent steps are
  merged into one task, which the sql_query lane answers with a single query (CTE), so no
  values have to be passed between tasks.
- **Plan check** (code): removes empty and duplicate tasks; more than 6 tasks sends the plan
  back to the planner once with the problem; a second failure falls back to the original
  question as one task.
- **run_task:** one full sql_query lane run per task, in parallel through LangGraph `Send`.

### 3.4 Data

Olist's public datasets, loaded into Postgres (Neon): a `raw` schema with the CSVs as text,
and a cleaned `analytics` schema of 10 typed tables with keys and a comment on every column:

| Table | Rows | Table | Rows |
| --- | ---: | --- | ---: |
| orders | 99,441 | order_reviews | 99,224 |
| order_items | 112,650 | products | 32,951 |
| order_payments | 103,886 | geolocation | 19,015 |
| customers | 96,096 | leads | 8,000 |
| sellers | 3,095 | closed_deals | 841 |

`catalog/build_catalog.py` reads the database into `catalog/schema_catalog.json` (columns,
types, descriptions, allowed values, ranges, joins, date facts), which every prompt uses.

### 3.5 Project structure

```
chatbot/          router, shared synthesizer, top-level graph (memory lives here)
analysis_lane/    planner, plan check, parallel task runner
sql_lane/         schema select, generate, review (code + LLM), execute, validate, graph
catalog/          schema catalog builder and generated catalog
db/               transform and check SQL for the analytics schema (see db/README.md)
eval/             golden set and DeepEval evaluations
tests/            smoke scripts for each graph
run_chatbot.py    terminal chat with a streamed answer
```

## 4. Evaluation

**Golden set:** `eval/sql_lane_questions.yaml`, 30 hand-verified questions: 9 single values,
8 lists (one genuinely empty result), 5 grouped, 4 time series and 4 questions the data cannot
answer (product names, profit, a forecast, personal details). Each has a reference SQL, the
expected result, the minimum tables, the expected shape and assumptions, and the data-model
traps it tests. Each trap was checked to actually change the answer when broken.

**Scoring compares results, not SQL text:** column names and order are ignored (extra
columns are fine), row order matters only when the question implies one, numbers match within
0.01, month labels are normalised.

| Eval | What it measures | Metrics (DeepEval) |
| --- | --- | --- |
| `eval/eval_schema_select.py` | Schema select alone | Table Recall (extra tables not penalised) |
| `eval/eval_generate.py` | One attempt: generate → review → execute → validate | Answerability, Pipeline Accuracy, SQL Result Match, Shape Match, Assumptions Quality (GEval), Refusal Quality (GEval); plus a funnel and a reviewer table (true catches, false rejections, misses) |
| `eval/eval_sql_pipeline.py` | The full sql_query lane, retries on | Final Accuracy, Schema Recall, Assumptions and Refusal Quality; plus first-attempt accuracy, retry recovery, failure breakdown, latency |

## 5. Results

| Metric | Result |
| --- | --- |
| **Final accuracy** | 83.3% |
| Schema recall (every needed table selected) | 100% |
| Silent wrong answers ("ok" but wrong result) | 3.8% |
| Latency per question | median 5.8 s |

## 6. Improvements

**Accuracy and review**
- Make the LLM-review value check look only at real `=`, `<>`, `IN` filters (never
  `ILIKE` patterns or queries without a filter), and accept a reviewer's rule check only when
  its quoted words really appear in the question. A first attempt at this was reverted after
  a lower score in one run, so it needs re-testing over several runs.
- Or move the exact-value check into the deterministic code review, where it can be checked
  exactly against the catalog's allowed values.
- Teach Generate to reduce `order_items` to distinct (order, seller/category) before attaching
  order-level values like review scores (q12), and to answer a filter value missing from the
  allowed list with an empty result instead of "unanswerable" (q17).
- Clarify the shape rule: "which X" returning the item and its value is a 1-row list, not a
  single value .

**Evaluation**
- Run each evaluation several times and report the mean and spread instead of one run.
- Add a baseline (plain gpt-4o with only the schema) to show how much the pipeline adds.
- Grow the golden set to 50-100 questions, partly written by someone other than the author.
- Add evaluations for the router (route accuracy, follow-up rewriting), the analysis planner
  and the synthesizer (faithfulness to the results).

**Chatbot and memory**
- Leave out-of-scope exchanges out of the 10-message history window, so small talk can't push
  real questions out of memory.
- A persistent checkpointer (e.g. Postgres) instead of in-memory, and a summary of older turns.
- A deterministic guard before the router, to refuse requests to change data without an LLM call.

**Product**
- Tasks that depend on each other's results (waves), passing values as SQL parameters.
- An API (FastAPI with streaming) and a web frontend with tables and charts.
- Token and cost tracking per question (currently in LangSmith traces), and caching of
  repeated questions.

## 7. Running it

**Requirements:** Python 3.11+, a Postgres database with the `analytics` schema (see
[`db/README.md`](db/README.md) to build it from the Olist CSVs), and an OpenAI API key.

```
pip install -r requirements.txt
```

Create a `.env` in the project root (never commit it):

```
DATABASE_URL="postgresql://<user>:<password>@<host>/<database>?sslmode=require"
OPENAI_API_KEY="<your key>"
```

Optional: `LANGSMITH_*` variables to trace every run in LangSmith.

**Chat in the terminal** (only the streamed answer is printed):

```
python run_chatbot.py
```

**Smoke scripts:**

```
python tests/sql_lane/test_sql_lane_graph.py
python tests/analysis_lane/test_analysis_lane_graph.py
python tests/chatbot/test_chatbot_graph.py
```

**Evaluations** (each costs OpenAI calls):

```
python eval/eval_schema_select.py
python eval/eval_generate.py
python eval/eval_sql_pipeline.py
```

On Windows, use `.venv\Scripts\python.exe` in place of `python`.

## 8. Data and licence

Data: Olist, [Brazilian E-Commerce Public Dataset by Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)
and [Marketing Funnel by Olist](https://www.kaggle.com/datasets/olistbr/marketing-funnel-olist),
published on Kaggle under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).
The cleaned `analytics` tables are a derivative of this data and are shared under the same
licence.
