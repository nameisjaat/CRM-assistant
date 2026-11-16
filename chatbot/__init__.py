"""The chatbot: a router that sends each question to the sql_query lane or the analysis
lane (or replies directly when it's out of scope), one synthesizer that writes the answer
for both lanes, and short-term memory of the conversation.
"""
