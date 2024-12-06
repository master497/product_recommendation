import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# # Hiển thị tiêu đề
# st.title("First Streamlit")
# st.write("## Chào mừng bạn đến với Streamlit")
# st.image("ham_spam.jpg", width=200)

# genre = st.radio(
#     "What's your gender?",
#     ["Male", "Female", "Other"],
#     index=None,
# )
# st.write("Your gender is:", genre)

import streamlit as st
tab1, tab2, tab3 = st.tabs(["Sản phẩm", "Người dùng", "Đánh giá"])
with tab1:
    st.header("Đề xuất sản phẩm")
    st.image("ham_spam.jpg", width=500)
with tab2:
    st.header("Đề xuất người dùng")
    st.image("ham_vs_spam.png", width=500)
with tab3:
    st.header("Đánh giá")
    st.write("Đánh giá của bạn về sản phẩm")
    st.write("Sản phẩm này thật là tốt!")
