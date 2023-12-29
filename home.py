import streamlit as st
import requests
from streamlit_lottie import st_lottie
from PIL import Image



def app():

    with st.container():
        lcol, rcol = st.columns((1,1))

        with lcol:
            st.write("")
            st.write("")
            st.write("")
            st.write("")
            st.write("")
            st.write("")
            imagePump = Image.open('C:\\Users\\Lannister\\OneDrive\\桌面\\web杨彬\\WbFig\\1.png')
            st.image(imagePump)
        with rcol:
            st.title('工程背景')
            st.write('  原油管道，作为国民经济的能源大动脉，承担着为我国经济发展提供坚实保障与推动的重任。在这其中，'
                     '泵被赞誉为原油管道的心脏，为油品流动提供源源不断动力。而在泵的运行过程中，泵前泵后压力的监测显得尤为重要，它是整个管道体系安全性至关重要的一环。')
            st.write('  为了更好地保障原油管道的安全与稳定运行，我们小组开展了一项关于基于差动电感传感器的弹簧管压力表的研究。'
                     '这种压力表凭借其差动电感传感器设计能够实现对压力的精确监测和稳定控制，从而为原油管道的安全运行提供有力支持。')

    with st.container():
        lcol, rcol = st.columns((1,.5))
        with lcol:
            st.write('__________')
            st.title('工程意义')
            st.write('工艺过程:使用压力表监测管段压力变化，通过差动电传感器，来传导信号，进行压力的实时监测。同时，给压力表设定个上下限，当压力表指针到达压力表所设定的下限，压力表指针与压力表下限的金属片，'
                '形成回路，同时启动泵，来进行增压的操作;''当压力增加到压力表所设定的上限时，压力表指针与压力表上限金属片，'
                '形成回路，进而停止泵继续运作，来阻止进一步的增压。')
            st.write('控制参数:压力，泵的操作状态(启停)')
            st.write('目的:针对管段的压力，进行实时监测，并通过压力表的监测，让管段压力处于一个合理的范围.'
                '在低压时，启动泵，来进行增压;在高压时，停泵，从而维持管段压力。')
        with rcol:
            st.write("")
            st.write("")
            st.write("")
            st.write("")
            st.write("")
            st.write("")
            st.write("")
            st.write("")
            st.write("")
            st.write("")
            st.write("")
            st.write("")
            st.write("")
            st.write("")
            imageFront = Image.open('C:\\Users\\Lannister\\OneDrive\\桌面\\web杨彬\\WbFig\\1.jpg')
            st.image(imageFront)