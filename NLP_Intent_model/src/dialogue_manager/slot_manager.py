class SlotManager:
    def __init__(self):
        self.slots = {
            "amount": None,
            "upi_id": None,
            "date": None,
            "account_type": None,
            "recipient": None,
            "phone_number": None,
            "credit_card_number": None,
            "loan_id": None,
            "bank_branch": None,
            # Added slots for loan queries:
            "loan_date": None,    # Corresponds to EMI due date or loan date
            "loan_amount": None,
            "loan_type": None,    # Optional: if you want specific loan types such as home, car
        }

    def fill_slots(self, entities):
        """Fill slots from entities. Allows overwriting existing values."""
        for ent, value in entities.items():
            if ent in self.slots:
                self.slots[ent] = value

    def missing_slots(self, required):
        """Return list of required slots that are missing."""
        return [slot for slot in required if self.slots.get(slot) is None]

    def get_filled_slots(self):
        """Return dict of currently filled slots."""
        return {k: v for k, v in self.slots.items() if v is not None}

    def reset(self):
        """Reset all slots to None."""
        for slot in self.slots:
            self.slots[slot] = None
    
    def get_slot(self, slot_name):
        """Get value of a specific slot."""
        return self.slots.get(slot_name)
