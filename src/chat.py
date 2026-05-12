"""Generation step.

Takes retrieved chunks + conversation history + the user question, builds a
strictly grounded system prompt, calls Gemini, and returns the answer and
citations. Enforces the 'answer only from context, else say I don't know' rule.
"""
