## Ayush Handoff – Speech ↔️ NLP Contract

This note captures everything Ayush (voice + infra) needs from the NLP service so he can plug ASR/TTS around it with zero guesswork during the demo.

### 1. Service Endpoints
- `POST /nlu`
  - Request: `{"text": "...", "top_k_intents": 3, "use_token_classifier": false}`
  - Response: `{"text": "...", "intents": [{"intent": "money_transfer", "confidence": 0.91}, ...], "entities": [{"entity": "amount", "value": "₹5000", "normalized": "5000"}]}`
- `POST /dialogue`
  - Request: `{"user_input": "...", "session_id": "<optional uuid>" }`
  - Response: `{"response": "Transferring 5000 to rahul@ybl.", "session_id": "<same uuid>" }`
- `POST /dialogue/reset`
  - Request: `{"session_id": "<optional uuid>" }`
  - Response: `{"status": "reset", "session_id": "<same uuid>" }`

Run with:
```bash
uvicorn src.integration.nlp_service:app --host 0.0.0.0 --port 8081 --reload
```

### 2. Data Handshake
1. Ayush streams user audio → ASR → `{ "text": "...", "language": "en-IN" }`.
2. He calls `/dialogue` with the `text` and a stable `session_id` (per UI tab or device).
3. NLP returns:
   ```json
   {
     "response": "₹500 has been successfully transferred to Rohan. Your new balance is ₹39,000.",
     "session_id": "abc-123"
   }
   ```
4. He feeds `response` into TTS + UI captions.

If he only needs raw intents/entities (e.g., experimenting with his own dialog layer) he can call `/nlu`.

### 3. Required Headers / Formats
- JSON UTF-8, `Content-Type: application/json`
- Keep numbers as plain numerals (e.g., 5000). NLP already normalizes amounts/UPI ids.
- Forward `language` if ASR detects it; multilingual routing happens later.

### 4. Error & Retry Signals
- `503` → NLP model still booting; retry with exponential backoff (Ayush can show “Warming up brain…” state).
- `409` → (future) OTP or auth requirements; for now, NLP sets `requires_auth` flag in the JSON it sends to backend (Utkarsh). Ayush only needs to bubble whatever DM tells him.
- Timeouts > 2s: Ayush should cut off mic animation and show “Still thinking…” while he keeps polling.

### 5. Ready-to-share Snippets
```
# Health
curl -X GET http://localhost:8081/health

# Dialogue
curl -X POST http://localhost:8081/dialogue ^
     -H "Content-Type: application/json" ^
     -d "{\"user_input\":\"Transfer 5000 rupees to rahul@ybl\",\"session_id\":\"demo-001\"}"

# NLU
curl -X POST http://localhost:8081/nlu ^
     -H "Content-Type: application/json" ^
     -d "{\"text\":\"Send 5k to rahul@ybl\",\"top_k_intents\":3}"
```

> **Tip for Ayush**  
> Cache the last assistant response. If he loses connection while the user is speaking, he can re-play the cached response with a “Repeating…” prefix once the NLP reply arrives.

### 6. Appendix – Sample Payloads

**Money Transfer (complete utterance)**
```json
Request:
{
  "user_input": "Transfer 5000 rupees to rahul@ybl",
  "session_id": "demo-001"
}

Response:
{
  "response": "Transferring 5000.0 to rahul@ybl.",
  "session_id": "demo-001"
}
```

**Multi-turn (slot filling)**
1. User: “I want to transfer money”
   - Response: “How much do you want to transfer?”
2. User: “5k”
   - Response: “To whom should I transfer? Please provide UPI ID.”
3. User: “rahul@ybl”
   - Response: “Transferring 5000.0 to rahul@ybl.”

**Fallback escalation**
```
Turn 1: “asdfghjkl” → “Sorry, I didn't understand that. Could you rephrase?”
Turn 2: “xyz abc” → “I'm not sure I understood. Can you try saying it differently?”
Turn 3: “random text” → “I'm still having trouble. Could you be more specific about what you need?”
```

### 7. Demo Handoff Checklist
- [x] Confirm he can hit `/health`
- [x] Share session IDs the UI will use (UUID4 strings are fine)
- [x] Provide 3 demo utterances + expected assistant replies
- [x] Agree on fallback visuals (“Could you rephrase?” overlays)
- [x] Schedule 15-min sync to walk him through the Postman collection

Once Ayush plugs this in, the full voice loop becomes:

`Mic (Tuhin) → ASR (Ayush) → /dialogue (Nandani) → reply JSON → TTS (Ayush) → UI playback (Tuhin)`

That’s all he needs for the Day-7 demo.

