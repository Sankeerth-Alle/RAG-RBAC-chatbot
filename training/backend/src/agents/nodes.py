import re
from pathlib import Path

from dotenv import load_dotenv

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chat_models import init_chat_model

from agents.prompts import custom_rag_prompt
from agents.states import State, AnswerWithSources


load_dotenv()


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[4]
VECTOR_DB_PATH = PROJECT_ROOT / "vector_store_db"

VALID_ACCESS_LEVELS = {
    "hr",
    "engineering",
    "c_level",
}


# ============================================================
# RETRIEVE
# ============================================================

def retrieve(state: State):

    question = state.get("question", "")
    access_level = state.get("access_level")

    # --------------------------------------------------------
    # 1. STRICT RBAC VALIDATION
    # --------------------------------------------------------

    if not isinstance(access_level, str):
        print(f"Invalid access level: {access_level!r}")
        return {"context": []}

    access_level = access_level.strip().lower()

    if access_level not in VALID_ACCESS_LEVELS:
        print(f"Invalid access level: {access_level!r}")
        return {"context": []}

    # --------------------------------------------------------
    # 2. INITIALIZE EMBEDDINGS
    # --------------------------------------------------------

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # --------------------------------------------------------
    # 3. CONNECT TO CORRECT CHROMA COLLECTION
    # --------------------------------------------------------

    vectorstore = Chroma(
        persist_directory=str(VECTOR_DB_PATH),
        collection_name="documentation",
        embedding_function=embeddings,
    )

    print(f"Using vector DB: {VECTOR_DB_PATH}")
    print(f"Access level: {access_level}")

    # --------------------------------------------------------
    # 4. COLLECTION INFO
    # --------------------------------------------------------

    try:
        collection_count = vectorstore._collection.count()
        print(f"Collection count: {collection_count}")
    except Exception:
        collection_count = "unknown"

    # --------------------------------------------------------
    # 5. FIRST: NORMAL SEMANTIC RETRIEVAL
    # --------------------------------------------------------

    retriever = vectorstore.as_retriever(
        search_kwargs={
            "k": 20,
            "filter": {
                "access_level": access_level
            },
        }
    )

    docs = retriever.invoke(question)

    print(f"Retrieved documents: {len(docs)}")

    # --------------------------------------------------------
    # 6. SECOND RBAC CHECK
    # --------------------------------------------------------

    authorized_docs = []

    for doc in docs:

        doc_access_level = doc.metadata.get("access_level")

        if (
            isinstance(doc_access_level, str)
            and doc_access_level.strip().lower() == access_level
        ):
            authorized_docs.append(doc)

    print(
        f"Authorized documents after metadata check: "
        f"{len(authorized_docs)}"
    )

    # --------------------------------------------------------
    # 7. EXACT NAME SEARCH FOR HR DATA
    # --------------------------------------------------------
    #
    # Semantic search is not reliable enough for structured
    # employee records.
    #
    # Example:
    #
    # "What is the salary of Aadhya Patel?"
    #
    # Gemini/embedding similarity may return other HR employees
    # instead of Aadhya Patel.
    #
    # Therefore, when the question contains a person's name,
    # search the authorized Chroma documents directly.
    # --------------------------------------------------------

    if access_level == "hr":

        # Extract a likely person name from common questions.
        name_patterns = [
            r"salary of ([A-Za-z]+(?:\s+[A-Za-z]+)+)",
            r"salary for ([A-Za-z]+(?:\s+[A-Za-z]+)+)",
            r"about ([A-Za-z]+(?:\s+[A-Za-z]+)+)",
            r"employee ([A-Za-z]+(?:\s+[A-Za-z]+)+)",
        ]

        candidate_name = None

        for pattern in name_patterns:

            match = re.search(
                pattern,
                question,
                flags=re.IGNORECASE
            )

            if match:
                candidate_name = match.group(1).strip()
                break

        if candidate_name:

            candidate_name_normalized = (
                candidate_name
                .lower()
                .strip()
                .replace("?", "")
                .replace(".", "")
            )

            print(
                f"Searching exact HR employee name: "
                f"{candidate_name}"
            )

            # ------------------------------------------------
            # Get all HR documents from the authorized
            # collection.
            #
            # IMPORTANT:
            # This is still RBAC-protected because we only
            # inspect documents whose metadata says "hr".
            # ------------------------------------------------

            try:

                all_hr_data = vectorstore._collection.get(
                    where={
                        "access_level": access_level
                    },
                    include=[
                        "documents",
                        "metadatas"
                    ],
                )

                documents = all_hr_data.get("documents", [])
                metadatas = all_hr_data.get("metadatas", [])

                exact_matches = []

                for content, metadata in zip(
                    documents,
                    metadatas
                ):

                    # Extra RBAC verification
                    metadata_access = metadata.get(
                        "access_level"
                    )

                    if (
                        not isinstance(
                            metadata_access,
                            str
                        )
                        or metadata_access.strip().lower()
                        != access_level
                    ):
                        continue

                    # Search the employee name inside the
                    # actual document content.
                    content_lower = content.lower()

                    if candidate_name_normalized in content_lower:

                        from langchain_core.documents import Document

                        exact_matches.append(
                            Document(
                                page_content=content,
                                metadata=metadata,
                            )
                        )

                if exact_matches:

                    print(
                        f"FOUND EXACT EMPLOYEE: "
                        f"{candidate_name}"
                    )

                    # For an exact employee query, return the
                    # matching employee records rather than
                    # unrelated semantic results.
                    authorized_docs = exact_matches

                    print(
                        f"Exact employee documents: "
                        f"{len(authorized_docs)}"
                    )

                else:

                    print(
                        f"Exact employee not found: "
                        f"{candidate_name}"
                    )

            except Exception as e:

                print(
                    f"Exact HR lookup failed: {e}"
                )

    # --------------------------------------------------------
    # 8. FINAL SAFETY CHECK
    # --------------------------------------------------------

    final_docs = []

    for doc in authorized_docs:

        doc_access_level = doc.metadata.get(
            "access_level"
        )

        if (
            isinstance(doc_access_level, str)
            and doc_access_level.strip().lower()
            == access_level
        ):
            final_docs.append(doc)

    print(
        f"Final authorized context documents: "
        f"{len(final_docs)}"
    )

    # --------------------------------------------------------
    # 9. RETURN CONTEXT
    # --------------------------------------------------------

    return {
        "context": final_docs
    }


# ============================================================
# GENERATE
# ============================================================

def generate(state: State):

    context = state.get("context", [])

    # --------------------------------------------------------
    # Build LLM context
    # --------------------------------------------------------

    docs_content = "\n\n".join(
        f"""
Context {idx + 1}

Metadata:
{doc.metadata}

Content:
{doc.page_content}
"""
        for idx, doc in enumerate(context)
    )

    # --------------------------------------------------------
    # If no authorized context exists
    # --------------------------------------------------------

    if not context:

        return {
            "answer": AnswerWithSources(
                sources=[],
                answer=(
                    "I don't know the answer because "
                    "the requested information is not "
                    "available for your access level."
                ),
            )
        }

    # --------------------------------------------------------
    # Create prompt
    # --------------------------------------------------------

    messages = custom_rag_prompt.invoke(
        {
            "question": state["question"],
            "context": docs_content,
        }
    )

    # --------------------------------------------------------
    # Initialize Gemini
    # --------------------------------------------------------

    model = init_chat_model(
        "gemini-3.1-flash-lite",
        model_provider="google_genai",
    )

    # --------------------------------------------------------
    # Structured output
    # --------------------------------------------------------

    model_structured_output = model.with_structured_output(
        AnswerWithSources
    )

    # --------------------------------------------------------
    # Generate answer
    # --------------------------------------------------------

    response = model_structured_output.invoke(
        messages
    )

    # --------------------------------------------------------
    # Return answer
    # --------------------------------------------------------

    return {
        "answer": response
    }
