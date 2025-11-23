# src/integration/mock_banking_api.py

MOCK_LOANS = {
    "hbcihsvd452": {
        "interest_rate": "9.5%",
        "remaining_amount": "45",
        "next_due": "today",
        "loan_id": "hbcihsvd452"
    },
    "jdjbd451": {
        "interest_rate": "10.2%",
        "remaining_amount": "785",
        "next_due": "tomorrow",
        "loan_id": "jdjbd451"
    },
    # Add more as needed...
}

def get_loan_details(loan_id):
    return MOCK_LOANS.get(loan_id, {
        "interest_rate": "Not available",
        "remaining_amount": "Not available",
        "next_due": "Not available",
        "loan_id": loan_id
    })
