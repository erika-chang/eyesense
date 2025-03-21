import streamlit as st

"# About the project 📚"

st.text("""
        Worldwide, about 2.2 million people have vision impairment. An early and efficient diagnosis tool could prevent about half of those cases.

        That's why we at Eyesense developed an AI-powered tool to help doctors diagnose the most common eye diseases using only one eye fundus image.

        Our models were trained using an ophthalmic database of 5,000 patients (right and left eye) and doctors' diagnostic keywords. This dataset represents patient information collected by Shanggong Medical Technology Co., Ltd. from different hospitals/medical centers in China.

        The model can classify the image into eight labels:
        -Normal (N)
        -Diabetes (D)
        -Glaucoma (G)
        -Cataract (C)
        -Age-related Macular Degeneration (A)
        -Hypertension (H)
        -Pathological Myopia (M)
        -Other diseases/abnormalities (O)
        """)
