import uvicorn
from fastapi import FastAPI, HTTPException, Depends, status, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field
from services import chatbot
from config import Config, validate_config
from auth import (
    LoginRequest,
    LoginResponse,
    authenticate_user,
    create_access_token,
    decode_access_token,
)
from dotenv import load_dotenv

load_dotenv()
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=Config.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Authorization", "Content-Type"],
)
security = HTTPBearer(auto_error=False)


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    message: str = Field(..., min_length=1, max_length=4000)


class ChatResponse(BaseModel):
    answer: str
    sources: list[str]
    user: dict[str, str]


@app.exception_handler(Exception)
async def handle_unexpected_error(request: Request, exc: Exception):
    """Return a concise 500 response while logging only the exception type."""
    print(f"ERROR: Unhandled API error on {request.url.path}: {type(exc).__name__}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Unable to process the request."}
    )

# ============================================================
# TOKEN-BASED AUTHENTICATION
# ============================================================

def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(security)
):
    """
    Extract and validate JWT token from Authorization header.
    
    Raises HTTPException with 403 status if token is invalid or expired.
    """
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = credentials.credentials
    
    payload = decode_access_token(token)
    if payload is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token"
        )
    
    return {"username": payload.username, "role": payload.role}


# ============================================================
# ENDPOINTS
# ============================================================

@app.post("/login", response_model=LoginResponse)
def login(request: LoginRequest):
    """
    Login endpoint - accepts username and password, returns JWT token.
    
    Request body:
        {
            "username": "Tony",
            "password": "your-password"
        }
    
    Response:
        {
            "access_token": "eyJ0eXAiOiJKV1QiLCJhbGc...",
            "token_type": "bearer",
            "username": "Tony",
            "role": "engineering"
        }
    
    Errors:
        - 401: Invalid username or password
    """
    user = authenticate_user(request.username, request.password)
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password"
        )
    
    # Generate JWT token
    token = create_access_token(user["username"], user["role"])
    
    return LoginResponse(
        access_token=token,
        token_type="bearer",
        username=user["username"],
        role=user["role"]
    )



# Protected test endpoint
@app.get("/test")
def test(user=Depends(get_current_user)):
    """
    Test endpoint to verify authentication is working.
    
    Requires: Authorization header with Bearer token
    
    Returns:
        {
            "message": "Hello {username}! You can now chat.",
            "role": "{role}"
        }
    """
    return {
        "message": f"Hello {user['username']}! You can now chat.",
        "role": user["role"]
    }


# Protected chat endpoint
@app.post("/chat", response_model=ChatResponse)
def query(request: ChatRequest, user=Depends(get_current_user)):
    """
    Chat endpoint - send message and get response from RAG chatbot.
    
    Requires: Authorization header with Bearer token
    
    Query parameters:
        - message (str): The question to ask the chatbot
    
    Returns:
        {
            "username": "{username}",
            "role": "{role}",
            "question": "{message}",
            "message": {answer object}
        }
    """
    message = request.message.strip()
    if not message:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Message cannot be empty"
        )

    answer = chatbot.chatbot_service(user["role"], message)
    return {
        "answer": answer.get("answer", "Unable to process the question."),
        "sources": answer.get("sources", []),
        "user": user
    }


@app.get("/")
def hello():
    """Health check endpoint."""
    return {"message": "Server is running.", "status": "ok"}


@app.get("/health")
def health():
    """Health check endpoint."""
    return {"status": "ok"}


def main():
    print("=" * 70)
    print("Starting RBAC-RAG Chatbot Backend")
    print("=" * 70)
    print()
    print(f"Configuration:")
    print(f"  Chat Model: {Config.GEMINI_CHAT_MODEL}")
    print(f"  Local Embedding Model: {Config.EMBEDDING_MODEL}")
    print(f"  Vector DB Path: {Config.VECTOR_DB_PATH}")
    print(f"  Server: {Config.BACKEND_HOST}:{Config.BACKEND_PORT}")
    print(f"  Authentication: JWT Bearer Token")
    print()

    # Validate configuration at startup
    issues = validate_config()
    if issues:
        print("Configuration Issues:")
        for issue in issues:
            print(f"  {issue}")
        
        # Check for critical errors (starting with ❌)
        errors = [
            issue for issue in issues
            if issue.startswith("❌") or issue.startswith("ERROR:")
        ]
        if errors:
            print()
            print("❌ CRITICAL ERRORS - Cannot start server. Fix the issues above.")
            print()
            return False
        print()
    
    # If validation passed, start the server
    print("=" * 70)
    print(f"Starting server on http://{Config.BACKEND_HOST}:{Config.BACKEND_PORT}")
    print("=" * 70)
    print()

    uvicorn.run(
        app,
        host=Config.BACKEND_HOST,
        port=Config.BACKEND_PORT
    )
    
    return True


if __name__ == "__main__":
    main()
