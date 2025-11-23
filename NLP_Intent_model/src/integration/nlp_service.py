"""
FastAPI service for NLP endpoints.
This will be fully implemented in Day 7, but provides a stub structure.
"""
from typing import Dict, List, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Import NLP components
try:
    from src.nlu.entity_extractor import combined_nlu
    from src.dialogue_manager.dialogue_manager import DialogueManager
except ImportError:
    # Fallback for testing
    combined_nlu = None
    DialogueManager = None


# Initialize FastAPI app
app = FastAPI(
    title="EchoFi NLP Service",
    description="NLP and Dialogue Management API for EchoFi Voice Assistant",
    version="1.0.0"
)

# Global dialogue manager instance
dialogue_manager = None


def get_dialogue_manager():
    """Get or create dialogue manager instance."""
    global dialogue_manager
    if dialogue_manager is None and DialogueManager is not None:
        dialogue_manager = DialogueManager()
    return dialogue_manager


# Request/Response models
class NLURequest(BaseModel):
    text: str
    top_k_intents: int = 1
    use_token_classifier: bool = False


class DialogueRequest(BaseModel):
    user_input: str
    session_id: Optional[str] = None


class NLUResponse(BaseModel):
    text: str
    intents: List[Dict]
    entities: List[Dict]


class DialogueResponse(BaseModel):
    response: str
    session_id: Optional[str] = None


# Health check endpoint
@app.get("/")
def root():
    """Root endpoint."""
    return {
        "service": "EchoFi NLP Service",
        "status": "running",
        "version": "1.0.0"
    }


@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "nlp"}


# NLU endpoint
@app.post("/nlu", response_model=NLUResponse)
async def nlu_endpoint(request: NLURequest):
    """
    NLU endpoint for intent classification and entity extraction.
    
    Args:
        request: NLURequest with text and options
    
    Returns:
        NLUResponse with intents and entities
    """
    if combined_nlu is None:
        raise HTTPException(status_code=503, detail="NLU service not available")
    
    try:
        result = combined_nlu(
            request.text,
            top_k_intents=request.top_k_intents,
            use_token_classifier=request.use_token_classifier
        )
        return NLUResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"NLU processing error: {str(e)}")


# Dialogue endpoint
@app.post("/dialogue", response_model=DialogueResponse)
async def dialogue_endpoint(request: DialogueRequest):
    """
    Dialogue management endpoint for conversational interactions.
    
    Args:
        request: DialogueRequest with user input and optional session_id
    
    Returns:
        DialogueResponse with assistant response
    """
    dm = get_dialogue_manager()
    if dm is None:
        raise HTTPException(status_code=503, detail="Dialogue manager not available")
    
    if combined_nlu is None:
        raise HTTPException(status_code=503, detail="NLU service not available")
    
    try:
        # Get NLU result
        nlu_result = combined_nlu(request.user_input)
        
        # Get response from dialogue manager
        response = dm.handle_turn(request.user_input, nlu_result)
        
        return DialogueResponse(
            response=response,
            session_id=request.session_id
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Dialogue processing error: {str(e)}")


# Reset session endpoint
@app.post("/dialogue/reset")
async def reset_session(session_id: Optional[str] = None):
    """Reset dialogue manager session."""
    dm = get_dialogue_manager()
    if dm is None:
        raise HTTPException(status_code=503, detail="Dialogue manager not available")
    
    try:
        dm.reset()
        return {"status": "reset", "session_id": session_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Reset error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

