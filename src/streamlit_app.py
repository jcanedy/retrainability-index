import streamlit as st

pg = st.navigation([
    st.Page("home.py", title="Home", icon=":material/house:"),
    st.Page("page1.py", title="Tier 2 Index", icon=":material/vital_signs:"),
    st.Page("page2.py", title="Tier 1 Index", icon=":material/owl:"),
])
pg.run()