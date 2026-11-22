"""MongoDB Atlas connection helpers for the Filmlytics Streamlit app.

The connection string is read from st.secrets["MONGO_URI"]:
  - locally:  Website/.streamlit/secrets.toml  (gitignored)
  - deployed: Streamlit Cloud -> Settings -> Secrets
Never hardcode the URI or commit it.
"""
import certifi
import streamlit as st
from pymongo import MongoClient
from pymongo.server_api import ServerApi


@st.cache_resource
def get_client() -> MongoClient:
    """Create one MongoClient per server process and reuse it across reruns."""
    return MongoClient(
        st.secrets["MONGO_URI"],
        server_api=ServerApi("1"),
        tlsCAFile=certifi.where(),
        serverSelectionTimeoutMS=8000,
    )


def ping() -> tuple[bool, str]:
    """Return (ok, message). Use this to confirm the connection works."""
    try:
        get_client().admin.command("ping")
        return True, "Connected to MongoDB Atlas."
    except KeyError:
        return False, 'MONGO_URI is missing from st.secrets (see Website/.streamlit/secrets.toml).'
    except Exception as e:  # network, auth, allowlist, DNS...
        return False, f"Connection failed: {e}"
