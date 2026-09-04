import streamlit as st
import requests
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

st.set_page_config(
    page_title="FinSolve Technologies Chatbot",
    page_icon="🤖",
)

# Initialize session state
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "token" not in st.session_state:
    st.session_state.token = None
if "username" not in st.session_state:
    st.session_state.username = ""
if "role" not in st.session_state:
    st.session_state.role = ""
if "messages" not in st.session_state:
    st.session_state.messages = []

# Get backend URL from environment or use default
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:6001")

def login_with_backend(username, password):
    try:
        response = requests.post(
            f"{BACKEND_URL}/login",
            json={"username": username, "password": password},
            timeout=10
        )
        if response.status_code != 200:
            return False, response.json().get("detail", "Login failed")

        data = response.json()
        st.session_state.token = data["access_token"]
        st.session_state.username = data["username"]
        st.session_state.role = data["role"]
        return True, ""
    except requests.RequestException:
        return False, "Cannot connect to backend server."


def authorization_headers():
    return {"Authorization": f"Bearer {st.session_state.token}"}


# Authentication page
if not st.session_state.authenticated:
    st.title("🔐 Login to FinSolve Technologies Chatbot")
    st.markdown("Please enter your credentials to access the chatbot.")

    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submit_button = st.form_submit_button("Login")

        if submit_button:
            if username and password:
                login_success, error_message = login_with_backend(username, password)
                if login_success:
                    st.session_state.authenticated = True
                    st.success("✅ Login successful! Redirecting...")
                    st.rerun()
                else:
                    st.error(f"❌ {error_message}")
            else:
                st.error("❌ Please enter both username and password.")

else:
    # Main chatbot interface
    # Header with logout option
    col1, col2 = st.columns([3, 1])
    with col1:
        st.title("🤖 FinSolve Technologies Chatbot")
        st.markdown(
            f"Welcome, **{st.session_state.username}**! Ask me anything about our knowledge base.")
    with col2:
        if st.button("🚪 Logout", type="secondary"):
            st.session_state.authenticated = False
            st.session_state.token = None
            st.session_state.username = ""
            st.session_state.role = ""
            st.session_state.messages = []
            st.rerun()

    # Function to call FastAPI backend with auth headers
    def get_bot_response(user_message):
        try:
            print("user message", user_message)

            response = requests.post(
                f"{BACKEND_URL}/chat",
                json={"message": user_message},
                headers=authorization_headers(),
                timeout=120
            )

            if response.status_code == 200:
                data = response.json()
                answer = data.get("answer", "Unable to process the question.")
                sources = data.get("sources", [])

                # Format the response with answer and sources
                formatted_response = answer
                if answer == "I don't have enough authorized information to answer that question.":
                    formatted_response = (
                        "**Access restricted**\n\n"
                        "You do not have permission to access this information."
                    )
                if sources:
                    formatted_response += f"\n\n**Sources:**\n"
                    for source in sources:
                        formatted_response += f"• {source}\n"

                return formatted_response
            elif response.status_code in (401, 403):
                st.error("❌ Session expired or invalid. Please login again.")
                st.session_state.authenticated = False
                st.session_state.token = None
                st.rerun()
            else:
                return f"Error: {response.json().get('detail', 'Request failed')}"

        except requests.exceptions.ConnectionError:
            return "❌ Cannot connect to the backend server. Please make sure it's running."
        except requests.exceptions.Timeout:
            return "⏱️ Request timed out. Please try again."
        except Exception as e:
            return f"❌ An error occurred: {str(e)}"

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask me anything..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get bot response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                bot_response = get_bot_response(prompt)
            st.markdown(bot_response)

        # Add bot response to chat history
        st.session_state.messages.append(
            {"role": "assistant", "content": bot_response})

    # Sidebar with additional options
    with st.sidebar:
        st.header("Chat Options")

        # User info
        st.info(
            f"👤 Logged in as: **{st.session_state.username}** "
            f"({st.session_state.role})"
        )

        # Clear chat button
        if st.button("🗑️ Clear Chat", type="secondary"):
            st.session_state.messages = []
            st.rerun()

        # Backend status check
        st.header("Backend Status")
        if st.button("🔍 Check Connection"):
            try:
                health_response = requests.get(
                    f"{BACKEND_URL}/health", timeout=5)
                if health_response.status_code == 200 and health_response.json().get("status") == "ok":
                    st.success("✅ Backend is running!")
                else:
                    st.error("❌ Backend is not responding correctly")
            except:
                st.error("❌ Cannot connect to backend")

        # Chat statistics
        st.header("Chat Statistics")
        st.metric("Messages sent", len(
            [msg for msg in st.session_state.messages if msg["role"] == "user"]))
        st.metric("Bot responses", len(
            [msg for msg in st.session_state.messages if msg["role"] == "assistant"]))
