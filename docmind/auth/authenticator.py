from pathlib import Path

import streamlit as st
import streamlit_authenticator as stauth
import yaml
from streamlit_authenticator.utilities.exceptions import LoginError, RegisterError
from yaml.loader import SafeLoader

authenticator_config_path = (Path(__file__).resolve().parent.parent.parent /
                             "authenticator_config.yaml")


class AuthenticatorConfig:
    def __init__(self, config_path=authenticator_config_path):
        self.config_path = Path(config_path)
        self.config = self.load_config()

    def load_config(self):
        try:
            with self.config_path.open('r', encoding='utf-8') as file:
                return yaml.load(file, Loader=SafeLoader)
        except FileNotFoundError:
            st.error("Configuration file not found.")
            raise

    def save_config(self):
        try:
            with self.config_path.open('w', encoding='utf-8') as file:
                yaml.dump(self.config, file, default_flow_style=False)
        except Exception as e:
            st.error(f"Failed to save configuration: {e}")


class Authenticator:
    def __init__(self, config_manager):
        self.config_manager = config_manager
        self.authenticator = stauth.Authenticate(
            self.config_manager.config['credentials'],
            self.config_manager.config['cookie']['name'],
            self.config_manager.config['cookie']['key'],
            self.config_manager.config['cookie']['expiry_days'],
            self.config_manager.config['pre-authorized']
        )

    def login(self):
        try:
            name, authentication_status, username = self.authenticator.login(
                fields={'Form name': 'Login', 'Username': 'Username', 'Password': 'Password', 'Login': 'Login'},
                location='main'
            )
            return name, authentication_status, username
        except LoginError as e:
            st.error(e)

    def register(self):
        try:
            return self.authenticator.register_user(
                location='main', pre_authorization=False, clear_on_submit=True
            )
            # return email, username, name
        except RegisterError as e:
            st.error(e)

    def update_user_details(self):
        """Update user details."""
        if st.session_state.get("authentication_status"):
            try:
                if self.authenticator.update_user_details(st.session_state["username"]):
                    st.success('Entries updated successfully')
            except stauth.UpdateError as e:
                st.error(f"Update failed: {e}")

    def logout(self, button_label: str = 'Logout', location: str = 'sidebar'):
        self.authenticator.logout(button_label, location)
