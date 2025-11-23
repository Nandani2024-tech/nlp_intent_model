# Quick Test Guide - EchoFi Voice Assistant

## 🚀 Quick Start

### Run All Test Scenarios
```bash
python test_conversation_scenarios.py
```

### Interactive Testing
```bash
python -m src.main
```

### Debug Mode (shows NLU details)
```bash
python -m src.main --debug
```

---

## 📋 Test Scenarios Included

### 1. **Normal Multi-turn Conversations**
- Money transfer with step-by-step slot filling
- Balance inquiry
- Complete information in one turn

### 2. **Fallback Scenarios**
- Random gibberish input
- Progressive fallback messages (3 levels)
- Recovery after fallbacks
- Maximum fallback limit

### 3. **Edge Cases**
- Empty input
- Whitespace only
- Special characters (₹, @, etc.)
- Abbreviations (5k, etc.)

### 4. **Intent Switching**
- Switching intents mid-conversation
- Context reset on new intent

### 5. **Context Persistence**
- Slots persist across turns
- Updating slot values
- Conversation state management

### 6. **Multilingual/Hinglish**
- Mixed language input
- Non-English phrases

---

## 🧪 Manual Test Cases

### Test Case 1: Basic Money Transfer
```
You: I want to transfer money
Expected: "How much do you want to transfer?"

You: 500 rupees
Expected: "To whom should I transfer? Please provide UPI ID."

You: rahul@ybl
Expected: "Transferring 500 to rahul@ybl."
```

### Test Case 2: Fallback Handling
```
You: asdfghjkl
Expected: "Sorry, I didn't understand that. Could you rephrase?"

You: xyz abc
Expected: "I'm not sure I understood. Can you try saying it differently?"

You: random text
Expected: "I'm still having trouble. Could you be more specific about what you need?"

You: check balance
Expected: Should work normally (fallback count reset)
```

### Test Case 3: Complete in One Turn
```
You: Transfer 1000 rupees to priya@paytm
Expected: "Transferring 1000 to priya@paytm."
```

### Test Case 4: Intent Switching
```
You: I want to transfer money
Expected: "How much do you want to transfer?"

You: Actually, check my balance instead
Expected: "Which account? Savings or current?" (resets transfer context)
```

### Test Case 5: Context Persistence
```
You: I want to transfer money
Expected: "How much do you want to transfer?"

You: 500
Expected: "To whom should I transfer? Please provide UPI ID."

You: Wait, make it 1000 instead
Expected: "To whom should I transfer? Please provide UPI ID." (amount updated to 1000)

You: suresh@ybl
Expected: "Transferring 1000 to suresh@ybl." (uses updated amount)
```

### Test Case 6: Balance Inquiry
```
You: What is my account balance?
Expected: "Which account? Savings or current?"

You: savings
Expected: "Your savings account balance is ₹82,500."
```

### Test Case 7: Special Characters
```
You: Transfer ₹5000 to friend@paytm
Expected: Should extract amount (5000) and UPI (friend@paytm)
```

### Test Case 8: Abbreviations
```
You: Transfer 5k to savings
Expected: Should recognize 5k as 5000
```

### Test Case 9 : Loan status

You : What is my current loan status (loan ID to use hbcihsvd452)
Expected : Could you share your loan ID to proceed?
---

## 🔍 What to Check

### ✅ Success Indicators
- [ ] Multi-turn conversations work smoothly
- [ ] Fallback messages are progressive and helpful
- [ ] Context persists across turns
- [ ] Slots are filled correctly
- [ ] Intent switching resets context properly
- [ ] Low confidence warnings appear when needed
- [ ] Conversation resets after completion

### ⚠️ Things to Watch For
- Low confidence scores (< 0.40) - may need more training
- Wrong intent predictions - check training data
- Missing entities - verify entity extraction
- Context not persisting - check dialogue manager state
- Fallback not triggering - check confidence threshold

---

## 🐛 Debugging Tips

### Enable Debug Mode
```bash
python -m src.main --debug
```
This shows:
- Detected intents with confidence scores
- Extracted entities
- NLU processing details

### Check Model Confidence
If you see low confidence warnings, the model might need:
- More training data
- More training epochs
- Better data quality

### Test Individual Components
```bash
# Test intent classification only
python test_intent_debug.py

# Test entity extraction
python -c "from src.nlu.entity_extractor import combined_nlu; print(combined_nlu('Transfer 500 to rahul@ybl'))"
```

---

## 📊 Expected Results

### Model Performance
- **Test Accuracy**: ~90% (from retraining output)
- **Confidence Threshold**: 0.25 (lowered for better UX)
- **Fallback Levels**: 3 progressive messages

### Response Times
- First request: ~1-2 seconds (model loading)
- Subsequent requests: < 0.5 seconds (cached model)

---

## 🎯 Next Steps

1. **Run the test scenarios**: `python test_conversation_scenarios.py`
2. **Try interactive mode**: `python -m src.main`
3. **Test with your own phrases**: Modify test scenarios
4. **Improve model**: Add more training examples if needed
5. **Adjust thresholds**: Modify confidence threshold in `dialogue_manager.py`

---

## 📝 Notes

- The model was retrained with 90% accuracy
- Confidence threshold is set to 0.25 (lowered from 0.40)
- Fallback handler has 3 progressive levels
- Model is cached after first load for faster inference

