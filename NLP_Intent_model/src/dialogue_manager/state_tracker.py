class StateTracker:
    def __init__(self):
        self.last_intent = None
        self.last_entities = {}
        self.history = []

    def update(self, intent, entities):
        self.last_intent = intent
        self.last_entities = entities
        self.history.append({"intent": intent, "entities": entities})

    def get_state(self):
        return {
            "intent": self.last_intent,
            "entities": self.last_entities
        }
