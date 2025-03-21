import streamlit as st

'''# Eyesense 👁️'''

st.sidebar.success('')

st.title("How to use Eyesense:")
st.text("Simply upload your image and click 'Predict' 🧙‍♀️ ")

image_file = st.file_uploader("Upload you image file here:", accept_multiple_files=False, type=['jpeg', 'png', 'jpg'])

if image_file:
    st.image(image_file)
