POLICY = {
    "money_transfer": {
    "required_slots": ["amount", "phone_number"],
    "success_message": "Transferring ₹{amount} to {phone_number_spelled}."
,
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
    "required_slots": ["loan_id"],
    "success_message": (
        "Your loan details: remaining amount is ₹{loan_amount}, "
        "loan ID {loan_id}, next EMI due on {loan_date}."
        ),
    },
    
    "set_reminder": {
        "required_slots": ["date"],
        "success_message": "I've set a reminder for {date}.",
    },
    "greeting": {
    "required_slots": [],
    "success_message": "Hello! How can I assist you today?"
    },
    "help": {
    "required_slots": [],
    "success_message": (
        "Sure! I can help with:\n"
        "- Money transfer\n"
        "- Checking balance\n"
        "- Loan status inquiries\n"
        "- Setting reminders\n\n"
        "Just tell me what you want to do."
    ),
}


}

