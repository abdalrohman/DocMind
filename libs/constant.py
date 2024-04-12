CHECKMARK_EMOJI = "✅"
THINKING_EMOJI = ":thinking_face:"
HISTORY_EMOJI = ":books:"
EXCEPTION_EMOJI = "⚠️"
ERROR_EMOJI = "❌"
HOME = "🏠"

SUPPORTED_PROVIDER = ["gemini_pro", "chat_fireworks", "groq", "openai"]

GEMINI_MODEL = [
    "gemini-1.5-pro-latest",
    "gemini-1.0-pro-latest",
]

FIREWORKS_MODEL = [
    "accounts/fireworks/models/firefunction-v1",
    "accounts/fireworks/models/mixtral-8x7b-instruct",
    "accounts/fireworks/models/yi-34b-200k-capybara",
    "accounts/fireworks/models/mixtral-8x7b",
    "accounts/fireworks/models/nous-hermes-2-mixtral-8x7b-dpo-fp8",
    "accounts/fireworks/models/mixtral-8x7b-instruct-hf",
]

GROQ_MODEL = ["mixtral-8x7b-32768", "llama2-70b-4096", "gemma-7b-it"]

OPENAI_MODEL = ["gpt-3.5-turbo-0125"]

SEARCH_ENGINES = {
    "google": {
        "api_keys": ["GOOGLE_SEARCH_API_KEY", "GOOGLE_CSE_ID"],
        "input_labels": ["Google API Key", "Google CSE ID"],
    },
    "exa": {"api_keys": ["EXA_API_KEY"], "input_labels": ["Exa API Key"]},
    "tavily": {"api_keys": ["TAVILY_API_KEY"], "input_labels": ["Tavily API Key"]},
}
