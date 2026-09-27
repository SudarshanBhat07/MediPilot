import streamlit as st

from rag import (
    extract_pdf_text,
    create_chunks,
    create_vector_index,
    retrieve_chunks,
    generate_answer,
    compare_documents
)


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="MediPilot",
    page_icon="🏥",
    layout="wide"
)


# =========================================================
# HEADER
# =========================================================

st.title("🏥 MediPilot")

st.subheader(
    "AI-Powered Healthcare Document Intelligence"
)

st.write(
    "Upload one or more healthcare documents "
    "to ask questions, continue conversations, "
    "or compare documents."
)


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.header("📄 Upload Documents")

    uploaded_files = st.file_uploader(
        "Choose healthcare PDFs",
        type=["pdf"],
        accept_multiple_files=True
    )

    st.divider()

    st.header("⚙️ Controls")

    if st.button(
        "🗑️ Clear Chat",
        use_container_width=True
    ):

        st.session_state.chat_history = []

        st.rerun()


# =========================================================
# CHECK FILE UPLOAD
# =========================================================

if not uploaded_files:

    st.info(
        "👈 Upload one or more PDF documents "
        "to get started."
    )

    st.stop()


# =========================================================
# CURRENT FILE NAMES
# =========================================================

current_file_names = [
    file.name
    for file in uploaded_files
]


# =========================================================
# SHOW UPLOADED DOCUMENTS
# =========================================================

st.success(
    f"📄 {len(uploaded_files)} document(s) uploaded."
)

with st.expander(
    "📂 View Uploaded Documents",
    expanded=False
):

    for file in uploaded_files:

        st.write(
            f"📄 {file.name}"
        )


# =========================================================
# INITIALIZE CHAT HISTORY
# =========================================================

if "chat_history" not in st.session_state:

    st.session_state.chat_history = []


# =========================================================
# PROCESS DOCUMENTS
# =========================================================

if (
    "processed_files" not in st.session_state
    or st.session_state.processed_files
    != current_file_names
):

    with st.spinner(
        "📖 Reading and processing documents..."
    ):

        all_documents = []

        # -----------------------------------------------
        # PROCESS EACH PDF
        # -----------------------------------------------

        for uploaded_file in uploaded_files:

            pages = extract_pdf_text(
                uploaded_file
            )

            if not pages:
                continue

            chunks = create_chunks(
                pages
            )

            # Add document name to every chunk
            for chunk in chunks:

                chunk["document"] = (
                    uploaded_file.name
                )

            all_documents.extend(
                chunks
            )


        # -----------------------------------------------
        # CHECK FOR READABLE CONTENT
        # -----------------------------------------------

        if not all_documents:

            st.error(
                "❌ No readable text was found "
                "in the uploaded PDF documents."
            )

            st.stop()


        # -----------------------------------------------
        # CREATE FAISS INDEX
        # -----------------------------------------------

        index = create_vector_index(
            all_documents
        )


        # -----------------------------------------------
        # STORE DATA IN SESSION
        # -----------------------------------------------

        st.session_state.chunks = (
            all_documents
        )

        st.session_state.index = (
            index
        )

        st.session_state.processed_files = (
            current_file_names
        )


        # New documents = new conversation
        st.session_state.chat_history = []


    st.success(
        f"✅ Successfully processed "
        f"{len(uploaded_files)} document(s)."
    )


# =========================================================
# MAIN DIVIDER
# =========================================================

st.divider()


# =========================================================
# SELECT OPERATION
# =========================================================

mode = st.radio(
    "Choose an operation",

    [
        "💬 Ask Documents",
        "🔄 Compare Documents"
    ],

    horizontal=True
)


# =========================================================
# ASK DOCUMENTS
# =========================================================

if mode == "💬 Ask Documents":

    st.header(
        "💬 Ask Your Documents"
    )

    st.caption(
        "MediPilot remembers the conversation "
        "during this session."
    )


    # =====================================================
    # DISPLAY CHAT HISTORY
    # =====================================================

    for chat in st.session_state.chat_history:

        with st.chat_message(
            chat["role"]
        ):

            st.write(
                chat["content"]
            )


    # =====================================================
    # CHAT INPUT
    # =====================================================

    question = st.chat_input(
        "Ask something about your healthcare documents..."
    )


    # =====================================================
    # PROCESS QUESTION
    # =====================================================

    if question:

        # -----------------------------------------------
        # SAVE CURRENT HISTORY BEFORE GENERATING
        # -----------------------------------------------

        previous_history = (
            st.session_state.chat_history.copy()
        )


        # -----------------------------------------------
        # DISPLAY USER QUESTION
        # -----------------------------------------------

        with st.chat_message(
            "user"
        ):

            st.write(
                question
            )


        # -----------------------------------------------
        # RETRIEVE RELEVANT CHUNKS
        # -----------------------------------------------

        with st.spinner(
            "🔎 Searching documents..."
        ):

            results = retrieve_chunks(

                question,

                st.session_state.chunks,

                st.session_state.index,

                top_k=6
            )


        # -----------------------------------------------
        # GENERATE ANSWER
        # -----------------------------------------------

        with st.spinner(
            "🤖 MediPilot is thinking..."
        ):

            answer = generate_answer(

                question,

                results,

                previous_history
            )


        # -----------------------------------------------
        # DISPLAY ANSWER
        # -----------------------------------------------

        with st.chat_message(
            "assistant"
        ):

            st.write(
                answer
            )


        # -----------------------------------------------
        # SAVE USER QUESTION
        # -----------------------------------------------

        st.session_state.chat_history.append(
            {
                "role": "user",
                "content": question
            }
        )


        # -----------------------------------------------
        # SAVE ASSISTANT ANSWER
        # -----------------------------------------------

        st.session_state.chat_history.append(
            {
                "role": "assistant",
                "content": answer
            }
        )


        # =================================================
        # SOURCES
        # =================================================

        if results:

            st.subheader(
                "📚 Sources"
            )

            source_pages = set()


            for result in results:

                document = result.get(
                    "document",
                    "Unknown document"
                )

                page = result.get(
                    "page",
                    "Unknown"
                )

                source_pages.add(
                    (
                        document,
                        page
                    )
                )


            for document, page in sorted(
                source_pages
            ):

                st.write(
                    f"📄 {document} — Page {page}"
                )


# =========================================================
# COMPARE DOCUMENTS
# =========================================================

else:

    st.header(
        "🔄 Compare Healthcare Documents"
    )


    # =====================================================
    # REQUIRE TWO DOCUMENTS
    # =====================================================

    if len(uploaded_files) < 2:

        st.warning(
            "⚠️ Please upload at least "
            "2 PDF documents to compare them."
        )


    else:

        st.success(
            f"Ready to compare "
            f"{len(uploaded_files)} documents."
        )


        st.write(
            "MediPilot will analyze the uploaded "
            "documents and identify similarities, "
            "differences, and important changes."
        )


        # =================================================
        # COMPARISON QUESTION
        # =================================================

        comparison_question = st.text_area(

            "What would you like to compare?",

            placeholder=(
                "Example: Compare the important "
                "information in these documents "
                "and identify the major differences."
            ),

            height=120
        )


        # =================================================
        # COMPARE BUTTON
        # =================================================

        if st.button(
            "🔄 Compare Documents",
            use_container_width=True
        ):


            # ---------------------------------------------
            # DEFAULT QUESTION
            # ---------------------------------------------

            if not comparison_question.strip():

                comparison_question = (
                    "Compare these documents and "
                    "identify the important "
                    "similarities and differences."
                )


            # ---------------------------------------------
            # RUN COMPARISON
            # ---------------------------------------------

            with st.spinner(
                "🔎 Comparing documents..."
            ):

                comparison = compare_documents(

                    comparison_question,

                    st.session_state.chunks
                )


            # ---------------------------------------------
            # DISPLAY RESULT
            # ---------------------------------------------

            st.subheader(
                "📊 Comparison Result"
            )

            st.write(
                comparison
            )


            # ---------------------------------------------
            # COMPARISON NOTE
            # ---------------------------------------------

            st.info(
                "The comparison is based on the "
                "uploaded documents."
            )


# =========================================================
# SAFETY NOTICE
# =========================================================

st.divider()

st.info(
    "🏥 MediPilot provides document-grounded "
    "healthcare information. It does not diagnose "
    "conditions, prescribe medicines, or replace "
    "professional medical advice."
)


# =========================================================
# FOOTER
# =========================================================

st.caption(
    "MediPilot • Gen AI + RAG Healthcare Assistant"
)