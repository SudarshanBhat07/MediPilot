import os
import time
import faiss
import numpy as np

from dotenv import load_dotenv
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from google import genai


# =========================================================
# CONFIGURATION
# =========================================================

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise ValueError(
        "GEMINI_API_KEY was not found in the .env file."
    )

client = genai.Client(
    api_key=API_KEY
)

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


# =========================================================
# PDF TEXT EXTRACTION
# =========================================================

def extract_pdf_text(pdf_file):

    reader = PdfReader(pdf_file)

    pages = []

    for page_number, page in enumerate(
        reader.pages,
        start=1
    ):

        text = page.extract_text()

        if text and text.strip():

            pages.append(
                {
                    "page": page_number,
                    "text": text.strip()
                }
            )

    return pages


# =========================================================
# CREATE TEXT CHUNKS
# =========================================================

def create_chunks(
    pages,
    chunk_size=500
):

    chunks = []

    for page in pages:

        words = page["text"].split()

        for i in range(
            0,
            len(words),
            chunk_size
        ):

            chunk_text = " ".join(
                words[
                    i:i + chunk_size
                ]
            )

            if chunk_text.strip():

                chunks.append(
                    {
                        "page": page["page"],
                        "text": chunk_text
                    }
                )

    return chunks


# =========================================================
# CREATE FAISS VECTOR INDEX
# =========================================================

def create_vector_index(chunks):

    if not chunks:
        raise ValueError(
            "No document chunks were found."
        )

    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    embeddings = embedding_model.encode(
        texts,
        convert_to_numpy=True
    )

    embeddings = np.asarray(
        embeddings,
        dtype="float32"
    )

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(
        dimension
    )

    index.add(
        embeddings
    )

    return index


# =========================================================
# RETRIEVE RELEVANT CHUNKS
# =========================================================

def retrieve_chunks(
    question,
    chunks,
    index,
    top_k=6
):

    if not chunks:
        return []

    question_embedding = (
        embedding_model.encode(
            [question],
            convert_to_numpy=True
        )
    )

    question_embedding = np.asarray(
        question_embedding,
        dtype="float32"
    )

    number_of_results = min(
        top_k,
        len(chunks)
    )

    distances, indices = index.search(
        question_embedding,
        number_of_results
    )

    results = []

    for idx in indices[0]:

        if idx < 0:
            continue

        results.append(
            chunks[idx]
        )

    return results


# =========================================================
# FORMAT CHAT HISTORY
# =========================================================

def format_chat_history(
    chat_history
):

    if not chat_history:

        return "No previous conversation."

    conversation = []

    for chat in chat_history:

        role = chat.get(
            "role",
            "unknown"
        )

        content = chat.get(
            "content",
            ""
        )

        if role == "user":

            conversation.append(
                f"User: {content}"
            )

        elif role == "assistant":

            conversation.append(
                f"MediPilot: {content}"
            )

    return "\n".join(
        conversation
    )


# =========================================================
# GEMINI RESPONSE
# =========================================================

def generate_gemini_response(
    prompt,
    max_attempts=3
):

    for attempt in range(
        max_attempts
    ):

        try:

            response = client.models.generate_content(

                model="gemini-3.8-flash",

                contents=prompt
            )

            return response.text


        except Exception as e:

            error_message = str(e)


            # =================================================
            # 429 - QUOTA EXCEEDED
            # =================================================

            if (
                "429" in error_message
                or "RESOURCE_EXHAUSTED" in error_message
            ):

                return (
                    "⚠️ Gemini API quota has been "
                    "exceeded for this project.\n\n"

                    "The current Gemini free-tier "
                    "request limit has been reached.\n\n"

                    "Please wait until the quota resets "
                    "or use a Gemini API project/plan "
                    "with additional available quota."
                )


            # =================================================
            # 503 - TEMPORARILY UNAVAILABLE
            # =================================================

            if (
                "503" in error_message
                or "UNAVAILABLE" in error_message
            ):

                if attempt < max_attempts - 1:

                    time.sleep(5)

                    continue

                return (
                    "⚠️ Gemini is temporarily "
                    "experiencing high demand.\n\n"

                    "Please wait a few seconds "
                    "and try again."
                )


            # =================================================
            # OTHER ERRORS
            # =================================================

            raise


    return (
        "⚠️ Unable to generate a response. "
        "Please try again."
    )


# =========================================================
# GENERATE CONTEXT-AWARE ANSWER
# =========================================================

def generate_answer(
    question,
    retrieved_chunks,
    chat_history=None
):

    if chat_history is None:

        chat_history = []


    # =====================================================
    # DOCUMENT CONTEXT
    # =====================================================

    context_parts = []

    for chunk in retrieved_chunks:

        document = chunk.get(
            "document",
            "Unknown document"
        )

        page = chunk.get(
            "page",
            "Unknown page"
        )

        text = chunk.get(
            "text",
            ""
        )

        context_parts.append(
            f"""
Document: {document}
Page: {page}

Content:
{text}
"""
        )


    document_context = "\n".join(
        context_parts
    )


    # =====================================================
    # PREVIOUS CONVERSATION
    # =====================================================

    previous_conversation = (
        format_chat_history(
            chat_history
        )
    )


    # =====================================================
    # GEMINI PROMPT
    # =====================================================

    prompt = f"""
You are MediPilot, a healthcare document
intelligence assistant.

Your task is to answer the user's current
question using the supplied healthcare
documents.

You also receive previous conversation history.
Use it to understand references and follow-up
questions.

==================================================
IMPORTANT RULES
==================================================

1. Use the supplied documents as the primary
   source of factual information.

2. Use previous conversation to understand
   references such as:

   - it
   - they
   - this
   - that
   - those
   - the first document
   - the second document
   - the previous answer

3. If the user asks a follow-up question,
   connect it to the previous conversation.

4. Do not invent information that is not
   supported by the supplied documents.

5. If the requested information cannot be
   found in the supplied documents, clearly
   say that it is not available in the
   uploaded documents.

6. Do not diagnose a person.

7. Do not prescribe medicines.

8. Do not recommend treatment changes.

9. Do not tell the user to stop or change
   prescribed medication.

10. Keep answers clear and easy to understand.

11. When possible, mention the document name
    and page number supporting the answer.

12. The uploaded documents are educational
    sources and should not replace advice
    from a qualified healthcare professional.

==================================================
PREVIOUS CONVERSATION
==================================================

{previous_conversation}

==================================================
RELEVANT DOCUMENT CONTENT
==================================================

{document_context}

==================================================
CURRENT USER QUESTION
==================================================

{question}

==================================================

Answer the current question.

Use the previous conversation to understand
context, but base factual claims primarily
on the supplied document content.
"""


    # =====================================================
    # GENERATE ANSWER
    # =====================================================

    return generate_gemini_response(
        prompt
    )


# =========================================================
# COMPARE DOCUMENTS
# =========================================================

def compare_documents(
    question,
    chunks
):

    documents = {}


    # =====================================================
    # GROUP CHUNKS BY DOCUMENT
    # =====================================================

    for chunk in chunks:

        document = chunk.get(
            "document",
            "Unknown document"
        )

        if document not in documents:

            documents[document] = []

        documents[document].append(
            chunk
        )


    # =====================================================
    # BUILD COMPARISON CONTEXT
    # =====================================================

    comparison_parts = []

    for (
        document_name,
        document_chunks
    ) in documents.items():

        comparison_parts.append(
            f"""
==================================================
DOCUMENT: {document_name}
==================================================
"""
        )

        for chunk in document_chunks:

            page = chunk.get(
                "page",
                "Unknown"
            )

            text = chunk.get(
                "text",
                ""
            )

            comparison_parts.append(
                f"""
Page: {page}

{text}
"""
            )


    comparison_context = "\n".join(
        comparison_parts
    )


    # =====================================================
    # COMPARISON PROMPT
    # =====================================================

    prompt = f"""
You are MediPilot, a healthcare document
comparison assistant.

Compare the uploaded healthcare documents
using ONLY the information provided in the
document content.

==================================================
USER'S COMPARISON REQUEST
==================================================

{question}

==================================================
DOCUMENT CONTENT
==================================================

{comparison_context}

==================================================
RESPONSE FORMAT
==================================================

Provide the comparison using these sections:

1. Summary

2. Similarities

3. Differences

4. Important Changes

5. Document and Page References

==================================================
IMPORTANT RULES
==================================================

- Use only the supplied documents.
- Do not invent information.
- Do not diagnose anyone.
- Do not prescribe treatment.
- Do not recommend medication changes.
- Clearly identify the document supporting
  important information.
- Include page numbers when available.
- If something cannot be determined from
  the documents, say so.
"""


    # =====================================================
    # GENERATE COMPARISON
    # =====================================================

    return generate_gemini_response(
        prompt
    )