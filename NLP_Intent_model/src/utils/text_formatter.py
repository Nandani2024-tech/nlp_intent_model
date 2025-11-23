# src/utils/text_formatter.py

DIGIT_WORDS = {
    '0': 'zero', '1': 'one', '2': 'two', '3': 'three', '4': 'four',
    '5': 'five', '6': 'six', '7': 'seven', '8': 'eight', '9': 'nine'
}

def spread_digits(number: str) -> str:
    """Return digits separated by spaces for clearer TTS."""
    return ' '.join(number)

def spell_out_digits(number: str) -> str:
    """Spell out each digit as a word for natural speech."""
    return ' '.join(DIGIT_WORDS.get(d, d) for d in number)
