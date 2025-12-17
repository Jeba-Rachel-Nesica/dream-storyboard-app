#!/usr/bin/env python
"""
Run the Streamlit app with proper Python path configuration
"""
import sys
import os

# Add project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Run streamlit
if __name__ == "__main__":
    import streamlit.web.cli as stcli
    import sys
    
    sys.argv = ["streamlit", "run", os.path.join(project_root, "app", "main.py")]
    sys.exit(stcli.main())
