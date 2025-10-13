import requests
import streamlit as st

st.write("Upload a file to FastAPI")
file = st.file_uploader("Choose a file", type=["pdf"])

if st.button("Submit"):
    if file is not None:
        files = {"file": (file.name, file, file.type)}
        response = requests.post("http://localhost:8000/file/upload", files=files)
        st.write(response)
    else:
        st.write("No file uploaded.")