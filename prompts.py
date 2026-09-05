"""System-prompt variants and grounding prompt templates for dataset generation."""

# Rotated across records so the model learns that system instructions actually
# steer behavior, rather than seeing one static prompt on every example.
SYSTEM_PROMPT_VARIANTS = [
    "You are a transportation engineering assistant. Answer clearly and precisely, "
    "using only the information you have been given.",
    "You are a senior transportation engineer mentoring a junior colleague. Explain "
    "concepts accurately and cite specifics (numbers, criteria, standards) when given.",
    "You are a concise technical reference assistant for transportation engineering. "
    "Give direct, factual answers without speculation.",
    "You are a transportation engineering instructor helping a student understand "
    "textbook material. Be accurate and avoid inventing facts not provided to you.",
]

GROUNDING_RULES = (
    "Use ONLY the information in the CONTEXT below to answer. Do not use outside "
    "knowledge, do not invent facts, names, dates, standards, or numbers that are not "
    "present in the CONTEXT. If the CONTEXT does not contain enough information to "
    "answer, respond with exactly: \"insufficient context\". Preserve hedging language "
    "from the CONTEXT (e.g. \"may\", \"could\", \"typically\") rather than turning it "
    "into an absolute statement. Do not invent author names or institutional affiliations."
)

QA_GENERATION_TEMPLATE = """{grounding_rules}

CONTEXT:
=== BEGIN CONTEXT ===
{context}
=== END CONTEXT ===

Task: Write ONE reading-comprehension question that is answerable solely from the CONTEXT \
above, and a grounded answer to that question using only facts stated in the CONTEXT.

Respond with ONLY a JSON object in this exact shape, no extra commentary:
{{"question": "...", "answer": "..."}}
"""

PARAPHRASE_TEMPLATE = """{grounding_rules}

Here is a question and its correct grounded answer, both derived from a technical context \
you cannot see directly, but must treat as authoritative:

QUESTION: {question}
ANSWER: {answer}

Task: Write ONE realistic, differently-phrased user question that a real person might type \
to ask for the SAME answer above. Vary wording, phrasing, and level of formality (a slightly \
messy or casual phrasing is fine), but keep it a genuine, sensible question. Do NOT change \
what is being asked, and do NOT change the answer.

Respond with ONLY a JSON object in this exact shape, no extra commentary:
{{"question": "..."}}
"""

TABLE_QA_GENERATION_TEMPLATE = """{grounding_rules}

The CONTEXT below is a data table extracted from a transportation engineering document.

=== BEGIN CONTEXT (TABLE) ===
{context}
=== END CONTEXT ===

Task: Write ONE question that asks about a specific value, row, or relationship in this \
table, and a grounded answer using only the data shown.

Respond with ONLY a JSON object in this exact shape, no extra commentary:
{{"question": "...", "answer": "..."}}
"""
