import streamlit as st
from multipage import MultiPage
import home,machinelearning
st.set_page_config(
        page_title=" ",
)
st.title("")
app = MultiPage()
app.add_page('首页',home.app)
app.add_page('机器学习模型构建',machinelearning.app)
if __name__=='__main__':
    app.run()