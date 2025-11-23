from .state_tracker import StateTracker
from .slot_manager import SlotManager
from .policy_rules import POLICY
from .fallback_handler import FallbackHandler
from src.integration.mock_banking_api import get_loan_details


class DialogueManager:
    def __init__(self):
        self.state = StateTracker()
        self.slots = SlotManager()
        self.fallback = FallbackHandler()
        self.current_intent = None  # Track active intent for continuous conversation
        self.conversation_active = False  # Track active multi-turn conversation

    def handle_turn(self, user_input: str, nlu_result: dict):
        user_input_lower = user_input.lower()
        intent_switch_keywords = ["actually", "instead", "cancel", "never mind", "nevermind", "change", "switch", "different"]
        is_intent_switch = any(keyword in user_input_lower for keyword in intent_switch_keywords)

        if is_intent_switch and self.conversation_active:
            self.slots.reset()
            self.conversation_active = False
            self.current_intent = None

        intents = nlu_result.get("intents", [])
        entities_list = nlu_result.get("entities", [])

        # -- KEY: If in a slot-filling conversation, and no intent is detected,
        # but entities are present, treat the input as active intent continuation:
        if self.conversation_active and self.current_intent and not intents and entities_list:
            intent = self.current_intent
            confidence = 0.75
        elif not intents:
            return self.fallback.handle_low_confidence()
        else:
            intent = intents[0].get("intent")
            confidence = intents[0].get("confidence", 0.0)

        if not intent:
            return self.fallback.handle_low_confidence()

        entities_dict = {}
        for ent in entities_list:
            ent_type = ent.get("entity")
            raw_value = ent.get("value")
            normalized = ent.get("normalized")
            if not ent_type:
                continue
            ent_value = normalized or raw_value
            if ent_value is None:
                continue

            # Amount merging logic
            if ent_type == "amount":
                existing = entities_dict.get("amount")
                try:
                    new_val = float(normalized) if normalized is not None else float(str(ent_value))
                except Exception:
                    new_val = None
                if existing is not None:
                    try:
                        existing_num = float(str(existing))
                    except Exception:
                        existing_num = None
                    if existing_num is None and new_val is not None:
                        entities_dict["amount"] = str(new_val)
                    elif existing_num is not None and new_val is not None and new_val > existing_num:
                        entities_dict["amount"] = str(new_val)
                else:
                    entities_dict["amount"] = str(new_val) if new_val is not None else ent_value
            else:
                entities_dict[ent_type] = ent_value

        # --- KEY: Map "date" to "loan_date" if in loan_query:
        if intent == "loan_query" and "date" in entities_dict:
            entities_dict["loan_date"] = entities_dict.pop("date")
        if intent == "loan_query" and "amount" in entities_dict:
            entities_dict["loan_amount"] = entities_dict.pop("amount")
        return self.handle_nlu_output(intent, entities_dict, confidence)



    def handle_nlu_output(self, intent, entities, confidence=1.0):
        if confidence < 0.40:
            self.fallback.increment_fallback_count()
            return self.fallback.handle_low_confidence(self.fallback.fallback_count)

        self.fallback.reset_fallback_count()

        if intent == "help":
            return (
                "Sure! Here’s what you can say:\n"
                "- \"Transfer 2000 to 9876543210\"\n"
                "- \"What's my savings balance?\"\n"
                "- \"Check my loan status\"\n"
                "- \"Set a reminder for tomorrow\"\n"
                "\nYou can continue anytime."
            )

        if intent == "greeting":
            self._reset_conversation()
            return "Hello! How can I assist you today?"

        if self.conversation_active and self.current_intent and intent != self.current_intent:
            if self._is_continuation(intent, entities):
                intent = self.current_intent
            else:
                self.slots.reset()
                self.conversation_active = False
                self.current_intent = None
        
        if intent == "loan_query" and "date" in entities and "loan_date" in self.slots.slots:
            entities["loan_date"] = entities.pop("date")

        self.state.update(intent, entities)
        self.slots.fill_slots(entities)

        # Auto-fill slots from mock db if loan_id is filled (and not already filled)
        if intent == "loan_query":
            loan_id = self.slots.get_slot("loan_id")
            if loan_id:
                details = get_loan_details(loan_id)
                if details["remaining_amount"] == "Not available":
                    msg = (
                        f"Sorry, no loan found with ID '{loan_id}'. "
                        "Please check your loan ID and try again."
                    )
                    self._reset_conversation()
                    return msg
                # Fill loan_amount if not already filled
                if self.slots.slots.get("loan_amount") is None:
                    self.slots.slots["loan_amount"] = details["remaining_amount"]
                # Fill loan_date if not already filled
                if self.slots.slots.get("loan_date") is None:
                    self.slots.slots["loan_date"] = details["next_due"]
                # You can add more slot auto-fill logic here if needed


        if not self.current_intent or intent != self.current_intent:
            self.current_intent = intent
            self.conversation_active = True

        policy_cfg = POLICY.get(intent, {})
        required = policy_cfg.get("required_slots", [])

        if not required and policy_cfg:
            msg = policy_cfg.get("success_message", "I have processed your request.")
            self._reset_conversation()
            return msg

        missing = self.slots.missing_slots(required)

        if missing:
            return self.fallback.handle_missing_slots(
                missing,
                current_intent=intent,
                filled_slots=self.slots.get_filled_slots()
            )

        try:
            if intent not in POLICY:
                filled_info = ", ".join([f"{k}: {v}" for k, v in self.slots.get_filled_slots().items()])
                msg = f"Processing {intent} with provided information: {filled_info}."
            elif intent == "loan_query":
                loan_id = self.slots.get_slot("loan_id")
                details = get_loan_details(loan_id)
                loan_amount = self.slots.get_slot("loan_amount") or details["remaining_amount"]
                loan_date = self.slots.get_slot("loan_date") or details["next_due"]
                spoken_loan_id = " ".join(details['loan_id'].upper())
                msg = (
                    f"Loan status for Loan I D {spoken_loan_id}. "
                    f"Remaining amount is rupees {loan_amount}. "
                    f"Interest rate is {details['interest_rate']} percent. "
                    f"Your next E M I is due on {loan_date}."
                )

            else:
                msg = POLICY[intent]["success_message"].format(**self.slots.slots)
        except KeyError:
            filled_info = ", ".join([f"{k}: {v}" for k, v in self.slots.get_filled_slots().items()])
            msg = f"Processing {intent} with provided information: {filled_info}."
        except Exception:
            msg = f"Processing {intent} with provided information."

        self._reset_conversation()
        return msg


    def _is_continuation(self, new_intent, entities):
        if not self.current_intent:
            return False

        required = POLICY.get(self.current_intent, {}).get("required_slots", [])
        missing = self.slots.missing_slots(required)

        if missing and entities:
            for slot in missing:
                if slot in entities:
                    return True

        return False

    def _reset_conversation(self):
        self.slots.reset()
        self.conversation_active = False
        self.current_intent = None

    def reset(self):
        self._reset_conversation()
        self.state = StateTracker()
        self.fallback.reset_fallback_count()
