"""
Main entry point for the NLP Intent Model service.
This file provides a simple CLI interface for testing the dialogue manager.
"""
import sys
from src.nlu.entity_extractor import combined_nlu
from src.dialogue_manager.dialogue_manager import DialogueManager


def main():
    """Main entry point for interactive conversation testing."""
    print("=" * 60)
    print("EchoFi Voice Assistant - NLP Service")
    print("=" * 60)
    print("Type 'exit' or 'quit' to end the conversation.\n")
    
    # Initialize dialogue manager
    dm = DialogueManager()
    
    while True:
        try:
            # Get user input
            user_input = input("You: ").strip()
            
            if user_input.lower() in ("exit", "quit", "bye"):
                print("\nThank you for using EchoFi Voice Assistant. Goodbye!")
                break
            
            if not user_input:
                continue
            
            # Get NLU result
            nlu_result = combined_nlu(user_input)
            
            # Debug: Show NLU output (optional)
            if "--debug" in sys.argv:
                print(f"\n[DEBUG] Intents: {nlu_result.get('intents', [])}")
                print(f"[DEBUG] Entities: {nlu_result.get('entities', [])}")
            
            # Always show low confidence warnings
            intents = nlu_result.get('intents', [])
            if intents and intents[0].get('confidence', 0) < 0.40:
                print(f"\n[WARNING] Low confidence: {intents[0].get('intent')} ({intents[0].get('confidence', 0):.3f})")
            
            # Get response from dialogue manager
            response = dm.handle_turn(user_input, nlu_result)
            
            # Print response
            print(f"Assistant: {response}\n")
            
        except KeyboardInterrupt:
            print("\n\nInterrupted by user. Goodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}")
            print("Please try again or type 'exit' to quit.\n")


if __name__ == "__main__":
    main()

