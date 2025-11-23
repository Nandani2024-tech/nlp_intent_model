"""
External API integration stubs for banking backend.
This module provides interfaces for integrating with backend banking APIs.
"""
from typing import Dict, Optional, Any
import logging

logger = logging.getLogger(__name__)


class BankingAPI:
    """Stub class for banking API integration."""
    
    def __init__(self, base_url: str = "https://api.bank.example.com"):
        """
        Initialize banking API client.
        
        Args:
            base_url: Base URL for banking API
        """
        self.base_url = base_url
        self.session = None  # Will be initialized with auth token
    
    def check_balance(self, account_type: str, user_id: str) -> Dict[str, Any]:
        """
        Check account balance.
        
        Args:
            account_type: Type of account (savings, current, etc.)
            user_id: User identifier
        
        Returns:
            Dict with balance information
        """
        # TODO: Implement actual API call in Day 7+
        logger.info(f"Checking {account_type} balance for user {user_id}")
        return {
            "account_type": account_type,
            "balance": 82500.00,
            "currency": "INR",
            "status": "success"
        }
    
    def transfer_funds(self, amount: float, phone_number: str, user_id: str) -> Dict[str, Any]:
        """
        Transfer funds to PHONE NUMBER.
        
        Args:
            amount: Transfer amount
            phone_number: Recipient Phone Number
            user_id: User identifier
        
        Returns:
            Dict with transfer status
        """
        # TODO: Implement actual API call in Day 7+
        logger.info(f"Transferring {amount} to {phone_number} for user {user_id}")
        return {
            "amount": amount,
            "recipient": phone_number,
            "transaction_id": "TXN123456789",
            "status": "success",
            "message": f"Successfully transferred ₹{amount} to {phone_number}"
        }
    
    def get_loan_info(self, loan_id: Optional[str] = None, user_id: str = None) -> Dict[str, Any]:
        """
        Get loan information.
        
        Args:
            loan_id: Optional loan ID
            user_id: User identifier
        
        Returns:
            Dict with loan information
        """
        # TODO: Implement actual API call in Day 7+
        logger.info(f"Getting loan info for user {user_id}, loan {loan_id}")
        return {
            "loan_id": loan_id or "LOAN001",
            "principal": 100000.00,
            "outstanding": 75000.00,
            "interest_rate": 8.5,
            "status": "active"
        }
    
    def set_reminder(self, date: str, description: str, user_id: str) -> Dict[str, Any]:
        """
        Set a reminder.
        
        Args:
            date: Reminder date
            description: Reminder description
            user_id: User identifier
        
        Returns:
            Dict with reminder status
        """
        # TODO: Implement actual API call in Day 7+
        logger.info(f"Setting reminder for {date}: {description} for user {user_id}")
        return {
            "reminder_id": "REM001",
            "date": date,
            "description": description,
            "status": "created"
        }
    
    def get_transactions(self, account_type: str, limit: int = 10, user_id: str = None) -> Dict[str, Any]:
        """
        Get recent transactions.
        
        Args:
            account_type: Type of account
            limit: Number of transactions to retrieve
            user_id: User identifier
        
        Returns:
            Dict with transaction list
        """
        # TODO: Implement actual API call in Day 7+
        logger.info(f"Getting {limit} transactions for {account_type} account of user {user_id}")
        return {
            "account_type": account_type,
            "transactions": [],
            "count": 0
        }


# Global banking API instance
_banking_api = None


def get_banking_api() -> BankingAPI:
    """Get or create banking API instance."""
    global _banking_api
    if _banking_api is None:
        _banking_api = BankingAPI()
    return _banking_api


def check_balance(account_type: str, user_id: str = "default") -> Dict[str, Any]:
    """Convenience function to check balance."""
    api = get_banking_api()
    return api.check_balance(account_type, user_id)


def transfer_funds(amount: float, phone_number: str, user_id: str = "default") -> Dict[str, Any]:
    api = get_banking_api()
    return api.transfer_funds(amount, phone_number, user_id)



def get_loan_info(loan_id: Optional[str] = None, user_id: str = "default") -> Dict[str, Any]:
    """Convenience function to get loan info."""
    api = get_banking_api()
    return api.get_loan_info(loan_id, user_id)


def set_reminder(date: str, description: str, user_id: str = "default") -> Dict[str, Any]:
    """Convenience function to set reminder."""
    api = get_banking_api()
    return api.set_reminder(date, description, user_id)

