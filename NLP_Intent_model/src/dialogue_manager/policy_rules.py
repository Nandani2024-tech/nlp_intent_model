POLICY = {
    "money_transfer": {
        "required_slots": ["amount", "upi_id"],
        "success_message": "Transferring {amount} to {upi_id}.",
    },
    "balance_inquiry": {
        "required_slots": ["account_type"],
        "success_message": "Your {account_type} account balance is ₹82,500.",
    },
    "check_balance": {  # Alias for balance_inquiry
        "required_slots": ["account_type"],
        "success_message": "Your {account_type} account balance is ₹82,500.",
    },
    "loan_query": {
    "required_slots": ["loan_id", "loan_date", "loan_amount"],
    "success_message": (
        "Your loan details: remaining amount is ₹{loan_amount}, "
        "loan ID {loan_id}, next EMI due on {loan_date}."
        ),
    },
    
    "set_reminder": {
        "required_slots": ["date"],
        "success_message": "I've set a reminder for {date}.",
    },
}
