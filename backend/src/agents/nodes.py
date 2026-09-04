import re
from pathlib import Path
from typing import List

from agents.prompts import custom_rag_prompt
from agents.states import State, AnswerWithSources
from config import Config, get_vector_db_path
from rbac import can_access_documents, is_valid_role

from langchain.chat_models import init_chat_model
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

from dotenv import load_dotenv

load_dotenv()


def generate(state: State):
    context = state.get("context", [])

    # If no authorized context exists, return early
    if not context:
        return {
            "answer": AnswerWithSources(
                answer=(
                    "I don't have enough authorized information to answer that question."
                ),
                sources=[]
            )
        }

    docs_content = "\n\n".join(
        f"Context {idx} (source: {doc.metadata.get('source_file', 'unknown')}):\n"
        f"{doc.page_content}"
        for idx, doc in enumerate(context)
    )

    messages = custom_rag_prompt.invoke(
        {
            "question": state["question"],
            "context": docs_content
        }
    )

    # Use configurable Gemini model
    try:
        model = init_chat_model(
            Config.GEMINI_CHAT_MODEL,
            model_provider="google_genai"
        )
    except Exception as e:
        print(f"Error initializing chat model {Config.GEMINI_CHAT_MODEL}: {e}")
        return {
            "answer": AnswerWithSources(
                answer="Sorry, I cannot generate a response due to an API error. Please try again later.",
                sources=[]
            )
        }

    model_structured_output = model.with_structured_output(
        AnswerWithSources
    )

    try:
        response = model_structured_output.invoke(messages)
        answer = response.get("answer", "") if isinstance(response, dict) else ""
        raw_sources = response.get("sources", []) if isinstance(response, dict) else []
        allowed_sources = {
            doc.metadata.get("source_file")
            for doc in context
            if doc.metadata.get("source_file")
        }
        sources = list(dict.fromkeys(
            source for source in raw_sources
            if isinstance(source, str) and source in allowed_sources
        ))
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError("Model returned an empty answer")
        return {
            "answer": AnswerWithSources(
                answer=answer.strip(),
                sources=sources
            )
        }
    except Exception as e:
        print(f"ERROR: Generation failed: {type(e).__name__}")
        return {
            "answer": AnswerWithSources(
                answer="Unable to process the question.",
                sources=[]
            )
        }


def retrieve(state: State):
    """
    Retrieve documents with strict RBAC validation.
    
    RBAC Flow:
    1. Validate input (type checking)
    2. Normalize access_level
    3. Verify against allowed roles
    4. Query vector DB with metadata filter
    5. Second RBAC check on retrieved docs
    6. Special handling for HR exact name matching
    7. Final safety check before return
    """

    question = state.get("question", "")
    access_level = state.get("access_level")

    if not isinstance(question, str) or not question.strip():
        print("ERROR: Empty question received")
        return {"context": []}

    # ============================================================
    # 1. STRICT RBAC VALIDATION - INPUT VALIDATION
    # ============================================================

    if not isinstance(access_level, str):
        print(f"ERROR: Invalid access level type: {type(access_level)}")
        return {
            "context": []
        }

    access_level = access_level.strip().lower()

    if not is_valid_role(access_level):
        print(f"ERROR: Invalid access level value: {access_level!r}")
        return {
            "context": []
        }

    # ============================================================
    # 2. INITIALIZE EMBEDDINGS AND VECTOR STORE
    # ============================================================

    try:
        embeddings = HuggingFaceEmbeddings(model_name=Config.EMBEDDING_MODEL)
    except Exception as e:
        print(f"ERROR: Failed to initialize embeddings: {e}")
        return {
            "context": []
        }

    db_path = str(get_vector_db_path())

    try:
        vector_store = Chroma(
            collection_name=Config.VECTOR_DB_COLLECTION_NAME,
            embedding_function=embeddings,
            persist_directory=db_path
        )
    except Exception as e:
        print(f"ERROR: Failed to connect to vector store at {db_path}: {e}")
        return {
            "context": []
        }

    collection_metadata = vector_store._collection.metadata or {}
    stored_embedding_model = collection_metadata.get("embedding_model")
    if stored_embedding_model != Config.EMBEDDING_MODEL:
        print(
            "ERROR: Vector collection embedding configuration is missing or "
            "incompatible; rebuild the vector database before querying"
        )
        return {"context": []}

    print(f"Access level: {access_level}")
    print(f"Using vector DB: {db_path}")

    # ============================================================
    # 3. FIRST RETRIEVAL - SEMANTIC SEARCH WITH METADATA FILTER
    # ============================================================

    try:
        # For c_level users, no metadata filter (access all docs)
        query_filter = None if access_level == "c_level" else {
            "access_level": access_level
        }

        retrieved_docs = vector_store.similarity_search(
            question,
            k=5,
            filter=query_filter
        )

        print(f"Retrieved {len(retrieved_docs)} documents from vector store")
    except Exception as e:
        print(f"ERROR: Vector store search failed: {e}")
        return {
            "context": []
        }

    # ============================================================
    # 4. SECOND RBAC CHECK - METADATA VALIDATION
    # ============================================================

    authorized_docs = []

    for doc in retrieved_docs:
        doc_access_level = doc.metadata.get("access_level")

        # Verify the document's access_level matches user's role
        if (
            isinstance(doc_access_level, str)
            and can_access_documents(
                access_level,
                doc_access_level.strip().lower()
            )
        ):
            authorized_docs.append(doc)

    print(f"Authorized documents after metadata check: {len(authorized_docs)}")

    # ============================================================
    # 5. HR EXACT NAME MATCHING (for structured employee data)
    # ============================================================

    if access_level == "hr":
        try:
            all_hr_data = vector_store._collection.get(
                where={"access_level": "hr"},
                include=["documents", "metadatas"]
            )
            documents = all_hr_data.get("documents", [])
            metadatas = all_hr_data.get("metadatas", [])
            question_lower = question.casefold()
            exact_matches = []

            for content, metadata in zip(documents, metadatas):
                metadata_access = str(metadata.get("access_level", "")).strip().lower()
                if not can_access_documents(access_level, metadata_access):
                    continue

                name_match = re.search(
                    r"(?:^|\n)full_name\s*:\s*([^\n]+)",
                    content,
                    flags=re.IGNORECASE
                )
                if not name_match:
                    name_match = re.search(
                        r"(?:^|,)full_name,([^,\n]+)",
                        content,
                        flags=re.IGNORECASE
                    )
                employee_name = name_match.group(1).strip() if name_match else ""
                if employee_name and re.search(
                    rf"(?<![A-Za-z]){re.escape(employee_name.casefold())}(?![A-Za-z])",
                    question_lower
                ):
                    exact_matches.append(Document(page_content=content, metadata=metadata))

            if exact_matches:
                authorized_docs = exact_matches
                print("Exact HR employee match found")
            else:
                print("No exact HR employee match; using semantic results")
        except Exception as e:
            print(f"ERROR: HR exact match search failed: {type(e).__name__}")

    # ============================================================
    # 6. FINAL SAFETY CHECK - TRIPLE VERIFICATION
    # ============================================================

    final_docs = []

    for doc in authorized_docs:
        doc_access_level = doc.metadata.get("access_level", "").strip().lower()

        # Final RBAC check
        if can_access_documents(access_level, doc_access_level):
            final_docs.append(doc)

    print(f"Final authorized documents: {len(final_docs)}")

    if not final_docs:
        print(f"No authorized documents found for access_level='{access_level}'")

    return {"context": final_docs}