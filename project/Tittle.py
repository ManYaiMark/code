import water as wt
import ussi_project as up
import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import time as t

# https://images.unsplash.com/photo-1495774539583-885e02cca8c2?ixlib=rb-4.0.3&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=1170&q=80
with st.spinner("loading...."):
    t.sleep(4.1)
st.sidebar.title("Menu")
select_box = st.sidebar.selectbox("Where are you going :",('intro', "water potability", "Prediction Prediction", "about web"))
if select_box == 'intro':
    st.markdown(
            f"""
                   <style>
                   .stApp {{
                       background-image: url("https://images.unsplash.com/photo-1507525428034-b723cf961d3e?ixlib=rb-4.0.3&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=1173&q=80");
                       background-attachment: fixed;
                       background-size: cover;
                       /* opacity: 0.3; */
                   }}
                   </style>
                   """,
            unsafe_allow_html=True
        )
        #     text = "**Water potability refers to the suitability of water for consumption by humans without causing any adverse health effects. Potable water must meet certain standards and guidelines established by regulatory bodies such as the World Health Organization (WHO) and the Environmental Protection Agency (EPA) to ensure its safety.\n\nThe quality of water can be affected by various factors, including natural contaminants such as minerals and microorganisms, as well as man-made pollutants such as chemicals and industrial waste. The presence of harmful substances in water can cause health problems such as gastrointestinal illness, reproductive problems, and even cancer.**\n\n**To ensure water potability, water treatment processes are used to remove contaminants and ensure that the water is safe for consumption. These processes may include filtration, disinfection, and the addition of chemicals to remove impurities.**"

    st.markdown("# Water Potability💧")


    st.write("**Water potability refers to the suitability of water for consumption by humans without causing any adverse health effects. Potable water must meet certain standards and guidelines established by regulatory bodies such as the World Health Organization (WHO) and the Environmental Protection Agency (EPA) to ensure its safety.**\n\n**The quality of water can be affected by various factors, including natural contaminants such as minerals and microorganisms, as well as man-made pollutants such as chemicals and industrial waste. The presence of harmful substances in water can cause health problems such as gastrointestinal illness, reproductive problems, and even cancer.**\n\n**To ensure water potability, water treatment processes are used to remove contaminants and ensure that the water is safe for consumption. These processes may include filtration, disinfection, and the addition of chemicals to remove impurities.**\n\n")
    st.write("**ความสามารถในการผลิตน้ำ หมายถึง ความเหมาะสมของน้ำสำหรับการบริโภคของมนุษย์โดยไม่ก่อให้เกิดผลเสียต่อสุขภาพ น้ำดื่มต้องเป็นไปตามมาตรฐานและหลักเกณฑ์ที่กำหนดโดยหน่วยงานกำกับดูแล เช่น องค์การอนามัยโลก (WHO) และสำนักงานคุ้มครองสิ่งแวดล้อม (EPA) เพื่อความปลอดภัย**\n\n**คุณภาพของน้ำอาจได้รับผลกระทบจาก ปัจจัยต่างๆ ได้แก่ สารปนเปื้อนจากธรรมชาติ เช่น แร่ธาตุและจุลินทรีย์ ตลอดจนสารมลพิษที่มนุษย์สร้างขึ้น เช่น สารเคมีและของเสียจากโรงงานอุตสาหกรรม การปรากฏตัวของสารอันตรายในน้ำอาจทำให้เกิดปัญหาสุขภาพ เช่น โรคระบบทางเดินอาหาร ปัญหาระบบสืบพันธุ์ และแม้แต่มะเร็ง**\n\n**เพื่อให้แน่ใจว่าน้ำดื่มได้ กระบวนการบำบัดน้ำจะใช้เพื่อกำจัดสิ่งปนเปื้อนและทำให้แน่ใจว่าน้ำมี ปลอดภัยต่อการบริโภค กระบวนการเหล่านี้อาจรวมถึงการกรอง การฆ่าเชื้อ และการเติมสารเคมีเพื่อขจัดสิ่งเจือปน**\n\n")

    data = pd.read_csv("./project/water_drop.csv")
    data = pd.DataFrame(data)
    x = data.drop(columns="Potability", axis=1)
    y = data["Potability"]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4)

    k_neighbors = np.arange(1, 51)
    train_score = np.empty(len(k_neighbors))
    test_score = np.empty(len(k_neighbors))

    for i, k in enumerate(k_neighbors):
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(x_train, y_train)
        train_score[i] = knn.score(x_train, y_train)
        test_score[i] = knn.score(x_test, y_test)

if select_box == "water potability":
    st.title("# Water Potability💧")
    wt.water()
if select_box == "Prediction Prediction":
    st.title(select_box)
    up.population()
if select_box == "about web":
    st.title(select_box)
    st.markdown("เว็ปนี้จัดทำขึ้นเพื่อศีกษาและหาข้อมูล streamlit ")
    st.write("จัดทำโดย นายอัษฎาวุธ  ประสารคำ 65114540710")
