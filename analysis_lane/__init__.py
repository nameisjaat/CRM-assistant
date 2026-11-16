"""The analysis lane: splits a complex question into independent sub-questions and answers
each one with the sql_query lane (unchanged, run in parallel). It returns the tasks and
their results; the chatbot's shared synthesizer writes the answer.

No waves: tasks never depend on each other's results. A dependent question is written as
one task, which the sql_query lane answers with a single query (a CTE).
"""
