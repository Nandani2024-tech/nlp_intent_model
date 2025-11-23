# Quick Test Guide - EchoFi Voice Assistant

*(Updated for Phone Number Support)*

## Quick Start

### Run All Test Scenarios

``` bash
python test_conversation_scenarios.py
```

### Interactive Testing

``` bash
python -m src.main
```

### Debug Mode (shows NLU details)

``` bash
python -m src.main --debug
```

## Test Scenarios Included

### 1. Normal Multi-turn Conversations

-   Money transfer with step-by-step slot filling using phone numbers
-   Balance inquiry
-   Complete information in one turn

### 2. Fallback Scenarios

-   Random gibberish input
-   Progressive fallback messages (3 levels)
-   Recovery after fallbacks
-   Maximum fallback limit

### 3. Edge Cases

-   Empty input
-   Whitespace only
-   Special characters (₹, @, etc.)
-   Abbreviations (5k, etc.)

### 4. Intent Switching

-   Switching intents mid-conversation
-   Context reset on new intent

### 5. Context Persistence

-   Slots persist across turns
-   Updating slot values (including amount and phone number)
-   Conversation state management

### 6. Multilingual / Hinglish

-   Mixed language input
-   Non-English phrases

## Manual Test Cases

### Test Case 1: Basic Money Transfer

You: I want to transfer money\
Expected: "How much do you want to transfer?"

You: 500 rupees\
Expected: "To whom should I transfer? Please provide the phone number."

You: 9876543210\
Expected: "Transferring 500 to 9876543210."

### Test Case 2: Fallback Handling

You: hmmmm\
Expected: "Sorry, I didn't understand that. Could you rephrase?"

You: huuuuu\
Expected: "I'm not sure I understood. Can you try saying it
differently?"

You: hehehehe\
Expected: "I'm still having trouble. Could you be more specific about
what you need?"

You: check balance\
Expected: Should work normally (fallback count reset)

### Test Case 3: Complete in One Turn

You: Transfer 1000 rupees to 9988776655\
Expected: "Transferring 1000 to 9988776655."

### Test Case 4: Intent Switching

You: I want to transfer money\
Expected: "How much do you want to transfer?"

You: Actually, check my balance instead\
Expected: "Which account? Savings or current?" (resets transfer context)

### Test Case 5: Context Persistence

You: I want to transfer money\
Expected: "How much do you want to transfer?"

You: 500\
Expected: "To whom should I transfer? Please provide the phone number."

You: Wait, make it 1000 instead\
Expected: "To whom should I transfer? Please provide the phone number."
(amount updated to 1000)

You: 9090909090\
Expected: "Transferring 1000 to 9090909090." (uses updated amount)

### Test Case 6: Balance Inquiry

You: What is my account balance?\
Expected: "Which account? Savings or current?"

You: Check my savings account balance\
Expected: "Your savings account balance is ₹82,500."

### Test Case 7: Special Characters

You: Transfer ₹5000 to 9876123456\
Expected: Should extract amount (5000) and phone_number (9876123456)

### Test Case 8: Abbreviations

You: Transfer 5k to 7899654123\
Expected: Should recognize 5k as 5000

### Test Case 9: Loan Status

You: What is my current loan status (loan ID to use hbcihsvd452)\
Expected: "Could you share your loan ID to proceed?"

You: How much loan do I still owe\
Expected: Shows remaining loan amount and next EMI due

## Enable Debug Mode

``` bash
python -m src.main --debug
```
