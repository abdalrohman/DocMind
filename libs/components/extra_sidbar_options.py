import json
from datetime import datetime
from pathlib import Path

import streamlit as st
from langchain.schema import messages_to_dict
from langchain_community.chat_message_histories.streamlit import StreamlitChatMessageHistory

from libs.constant import CHECKMARK_EMOJI


class HistoryExporter:
    def __init__(
            self,
            history
            ):
        self.history = history
        self.messages_dict = messages_to_dict(history.messages)
        self.export_history_dir = Path("./data/export_history")
        self.export_history_dir.mkdir(parents=True, exist_ok=True)

    def _export(
            self,
            file_path,
            data,
            format_func=str
            ):
        with open(file_path, 'w') as f:
            f.write(format_func(data))

    def export_history(
            self,
            format,
            mime_type
            ):
        file_path = self.export_history_dir / f"chat_history-{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.{format}"
        data = self.history if format == 'txt' else self.messages_dict
        format_func = str if format == 'txt' else json.dumps
        self._export(file_path, data, format_func)
        st.download_button(
            label=f"Download {format} history",
            data=format_func(data),
            file_name=f"chat_history-{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.{format}",
            mime=mime_type,
            use_container_width=True
            )


def additional_options(
        history: StreamlitChatMessageHistory
        ):
    exporter = HistoryExporter(history)

    # Sidebar options
    with st.sidebar:
        with st.popover("Session state:", use_container_width=True):
            state = dict(sorted(st.session_state.items()))
            st.write(state)

        # Clear and export history buttons
        clear_button, export_button = st.columns([1, 1], gap="small")
        if clear_button.button("Clear History", use_container_width=True):
            st.session_state.clear()
            st.rerun()
        if export_button.button("Export history", use_container_width=True):
            exporter.export_history('txt', 'text/plain')
            exporter.export_history('json', 'application/json')
            st.toast(f"{CHECKMARK_EMOJI} History exported.")
