"""
Security utilities for user service.
"""
import hashlib
import secrets
import re
import bcrypt


class PasswordHasher:
    """Password hashing utility using bcrypt."""
    
    def hash_password(self, password: str) -> str:
        """Hash a password using bcrypt."""
        # Convert to bytes and truncate if necessary
        password_bytes = password.encode('utf-8')[:72]
        salt = bcrypt.gensalt()
        return bcrypt.hashpw(password_bytes, salt).decode('utf-8')
    
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify a password against its hash."""
        try:
            password_bytes = plain_password.encode('utf-8')[:72]
            hashed_bytes = hashed_password.encode('utf-8')
            return bcrypt.checkpw(password_bytes, hashed_bytes)
        except Exception:
            return False
    
    def needs_update(self, hashed_password: str) -> bool:
        """Check if password hash needs to be updated."""
        # For simplicity, always return False
        return False


class PasswordValidator:
    """Password strength validation."""
    
    @staticmethod
    def validate_password_strength(password: str) -> tuple[bool, list[str]]:
        """
        Validate password strength.
        
        Returns:
            tuple: (is_valid, list_of_errors)
        """
        errors = []
        
        if len(password) < 8:
            errors.append("Password must be at least 8 characters long")
        
        if len(password) > 128:
            errors.append("Password must be less than 128 characters long")
        
        if not re.search(r"[A-Z]", password):
            errors.append("Password must contain at least one uppercase letter")
        
        if not re.search(r"[a-z]", password):
            errors.append("Password must contain at least one lowercase letter")
        
        if not re.search(r"\d", password):
            errors.append("Password must contain at least one digit")
        
        if not re.search(r"[!@#$%^&*()_+\-=\[\]{}|;:,.<>?]", password):
            errors.append("Password must contain at least one special character")
        
        # Check for common patterns
        if password.lower() in ["password", "123456", "qwerty", "admin"]:
            errors.append("Password is too common")
        
        return len(errors) == 0, errors


class EmailValidator:
    """Email validation utility."""
    
    @staticmethod
    def validate_email_format(email: str) -> bool:
        """Validate email format using regex."""
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return re.match(pattern, email) is not None
    
    @staticmethod
    def normalize_email(email: str) -> str:
        """Normalize email address (lowercase, strip whitespace)."""
        return email.strip().lower()


class TokenGenerator:
    """Secure token generation utility."""
    
    @staticmethod
    def generate_reset_token() -> str:
        """Generate a secure password reset token."""
        return secrets.token_urlsafe(32)
    
    @staticmethod
    def generate_verification_token() -> str:
        """Generate a secure email verification token."""
        return secrets.token_urlsafe(24)
    
    @staticmethod
    def generate_api_key() -> str:
        """Generate a secure API key."""
        return secrets.token_urlsafe(40)


# Global instances
password_hasher = PasswordHasher()
password_validator = PasswordValidator()
email_validator = EmailValidator()
token_generator = TokenGenerator()