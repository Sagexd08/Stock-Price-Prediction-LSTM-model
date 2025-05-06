"""
Authentication Module

This module provides functions for user authentication in the Streamlit app.
"""

import os
import json
import logging
import streamlit as st
import hmac
import hashlib
import base64
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Try to import Firebase modules, with graceful fallback if not available
try:
    import firebase_admin
    from firebase_admin import auth
    FIREBASE_AVAILABLE = True
except ImportError:
    logger.warning("Firebase auth module not available. Using local authentication.")
    FIREBASE_AVAILABLE = False

class AuthManager:
    """
    Class for managing user authentication.
    """
    def __init__(self):
        """
        Initialize the authentication manager.
        """
        self.initialized = FIREBASE_AVAILABLE
        
        # Load local user credentials if Firebase is not available
        if not self.initialized:
            self.users = self._load_local_users()
    
    def _load_local_users(self):
        """
        Load user credentials from local file.
        """
        try:
            users_file = os.path.join('data', 'users.json')
            
            if os.path.exists(users_file):
                with open(users_file, 'r') as f:
                    return json.load(f)
            else:
                # Create default users file
                default_users = {
                    "admin": {
                        "password": self._hash_password("admin123"),
                        "name": "Admin User",
                        "email": "admin@example.com",
                        "role": "admin"
                    },
                    "user": {
                        "password": self._hash_password("user123"),
                        "name": "Demo User",
                        "email": "user@example.com",
                        "role": "user"
                    }
                }
                
                # Create directory if it doesn't exist
                os.makedirs(os.path.dirname(users_file), exist_ok=True)
                
                # Save default users
                with open(users_file, 'w') as f:
                    json.dump(default_users, f, indent=4)
                
                return default_users
                
        except Exception as e:
            logger.error(f"Error loading local users: {str(e)}")
            return {}
    
    def _hash_password(self, password):
        """
        Hash a password for storing.
        """
        salt = "stockprediction"  # In a real app, use a unique salt per user
        return hashlib.sha256((password + salt).encode()).hexdigest()
    
    def _verify_password(self, stored_password, provided_password):
        """
        Verify a stored password against a provided password.
        """
        return stored_password == self._hash_password(provided_password)
    
    def login(self, username, password):
        """
        Authenticate a user.
        
        Parameters:
        -----------
        username : str
            Username
        password : str
            Password
        
        Returns:
        --------
        dict or None
            User info if authentication successful, None otherwise
        """
        if self.initialized:
            # Use Firebase authentication
            try:
                # In a real app, you would use Firebase Auth REST API or SDK
                # For this example, we'll fall back to local authentication
                return self._local_login(username, password)
            except Exception as e:
                logger.error(f"Error with Firebase authentication: {str(e)}")
                return self._local_login(username, password)
        else:
            # Use local authentication
            return self._local_login(username, password)
    
    def _local_login(self, username, password):
        """
        Authenticate a user using local credentials.
        """
        if username in self.users and self._verify_password(self.users[username]['password'], password):
            user_info = {
                'username': username,
                'name': self.users[username].get('name', username),
                'email': self.users[username].get('email', ''),
                'role': self.users[username].get('role', 'user')
            }
            return user_info
        return None
    
    def create_session(self, user_info):
        """
        Create a session for an authenticated user.
        
        Parameters:
        -----------
        user_info : dict
            User information
        """
        # Store user info in session state
        st.session_state['user'] = user_info
        st.session_state['authenticated'] = True
        st.session_state['login_time'] = datetime.now()
        
        # Create a session token
        token = self._create_session_token(user_info)
        st.session_state['token'] = token
    
    def _create_session_token(self, user_info):
        """
        Create a session token.
        """
        # In a real app, use a proper JWT library
        # For this example, we'll use a simple base64 encoding
        payload = {
            'username': user_info['username'],
            'role': user_info['role'],
            'exp': (datetime.now() + timedelta(hours=24)).timestamp()
        }
        
        # Convert to JSON and encode
        token = base64.b64encode(json.dumps(payload).encode()).decode()
        return token
    
    def logout(self):
        """
        Log out the current user.
        """
        if 'user' in st.session_state:
            del st.session_state['user']
        
        if 'authenticated' in st.session_state:
            del st.session_state['authenticated']
        
        if 'token' in st.session_state:
            del st.session_state['token']
        
        if 'login_time' in st.session_state:
            del st.session_state['login_time']
    
    def is_authenticated(self):
        """
        Check if the current user is authenticated.
        
        Returns:
        --------
        bool
            True if authenticated, False otherwise
        """
        if 'authenticated' not in st.session_state or not st.session_state['authenticated']:
            return False
        
        # Check if session has expired
        if 'login_time' in st.session_state:
            login_time = st.session_state['login_time']
            if datetime.now() - login_time > timedelta(hours=24):
                self.logout()
                return False
        
        return True
    
    def get_current_user(self):
        """
        Get the current authenticated user.
        
        Returns:
        --------
        dict or None
            User info if authenticated, None otherwise
        """
        if self.is_authenticated() and 'user' in st.session_state:
            return st.session_state['user']
        return None
    
    def require_auth(self):
        """
        Require authentication to access the app.
        
        Returns:
        --------
        bool
            True if authenticated, False otherwise
        """
        if not self.is_authenticated():
            self._show_login_form()
            return False
        return True
    
    def _show_login_form(self):
        """
        Show a login form.
        """
        st.markdown("""
        <div style="background: linear-gradient(90deg, #1e3a8a 0%, #4e8df5 100%); padding:20px; border-radius:12px; box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);">
            <h1 style="color:white; text-align:center; margin:0; text-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);">📈 Stock Price Prediction Dashboard</h1>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div style="background: linear-gradient(to right, #f0f2f6, #ffffff); padding:20px; border-radius:8px; margin-top:20px; margin-bottom:25px; box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);">
            <p style="font-size:16px; color:#1e3a8a; text-align:center; line-height:1.6;">
                Please log in to access the dashboard.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            with st.form("login_form"):
                username = st.text_input("Username")
                password = st.text_input("Password", type="password")
                
                if st.form_submit_button("Login"):
                    user_info = self.login(username, password)
                    
                    if user_info:
                        self.create_session(user_info)
                        st.rerun()
                    else:
                        st.error("Invalid username or password")
            
            st.markdown("""
            <div style="text-align:center; margin-top:20px; font-size:14px; color:#4b5563;">
                <p>Default credentials:</p>
                <p>Username: <code>user</code> | Password: <code>user123</code></p>
                <p>Admin: <code>admin</code> | Password: <code>admin123</code></p>
            </div>
            """, unsafe_allow_html=True)

# Initialize authentication manager
auth_manager = AuthManager()

# Export the instance for use in other modules
def get_auth_manager():
    """
    Get the authentication manager instance.
    
    Returns:
    --------
    AuthManager
        Authentication manager instance
    """
    return auth_manager
