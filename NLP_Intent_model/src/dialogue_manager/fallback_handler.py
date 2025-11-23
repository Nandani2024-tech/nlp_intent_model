class FallbackHandler:
    def __init__(self):
        self.fallback_count = 0
        self.max_fallbacks = 3
    
    def handle_missing_slots(self, missing, current_intent=None, filled_slots=None):
        """
        Handle missing slots with context-aware prompts.
        
        Args:
            missing: List of missing slot names
            current_intent: Current intent being processed
            filled_slots: Dict of already filled slots for context
        """
        filled_slots = filled_slots or {}
        
        # Context-aware prompts based on intent and what's already filled
        prompts = {
            "amount": {
                "default": "How much do you want to transfer?",
                "with_context": "What amount would you like to transfer?"
            },
            "phone_number": {
                "default": "Please share the phone number of the recipient.",
                "with_context": "To whom should I transfer? Please provide the phone number."
            },

            "date": {
                "default": "On which date?",
                "with_context": "When would you like this reminder set?"
            },
            "account_type": {
                "default": "Which account? Savings or current?",
                "with_context": "For which account would you like to check the balance?"
            },
            # New loan-related slots:
            "loan_id": {
                "default": "Please provide your loan ID.",
                "with_context": "Could you share your loan ID to proceed?"
            },
            "loan_date": {  # alternative key if used
                "default": "When is your next EMI due?",
                "with_context": "Please tell me the EMI due date."
            },
            "loan_amount": {
                "default": "What is the remaining loan amount?",
                "with_context": "How much loan amount is left?"
            }
        }
        
        slot = missing[0]
        prompt_config = prompts.get(slot, {"default": "Can you provide more details?", "with_context": "I need more information."})
        
        # Use context-aware prompt if we have filled slots
        if filled_slots:
            return prompt_config.get("with_context", prompt_config["default"])
        else:
            return prompt_config["default"]
    
    def handle_low_confidence(self, fallback_count=0):
        """
        Handle low confidence with progressive fallback messages.
        
        Args:
            fallback_count: Number of consecutive fallbacks
        """
        if fallback_count >= self.max_fallbacks:
            return "I'm having trouble understanding. Could you please rephrase your request differently, or say 'help' for assistance?"
        
        fallback_messages = [
            "Sorry, I didn't understand that. Could you rephrase?",
            "I'm not sure I understood. Can you try saying it differently?",
            "I'm still having trouble. Could you be more specific about what you need?"
        ]
        
        idx = min(fallback_count, len(fallback_messages) - 1)
        return fallback_messages[idx]
    
    def increment_fallback_count(self):
        """Increment fallback counter."""
        self.fallback_count += 1
    
    def reset_fallback_count(self):
        """Reset fallback counter on successful understanding."""
        self.fallback_count = 0
