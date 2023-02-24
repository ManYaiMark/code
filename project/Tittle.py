import water as wt
import ussi_project as up
import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import time as t

with st.spinner("loading...."):
    t.sleep(4.1)
st.sidebar.title("Menu")
select_box=st.sidebar.selectbox("Where are you going :",('intro',"water potability","Prediction Prediction","about web"))
if select_box == 'intro':
    text = "**Water potability refers to the suitability of water for consumption by humans without causing any adverse health effects. Potable water must meet certain standards and guidelines established by regulatory bodies such as the World Health Organization (WHO) and the Environmental Protection Agency (EPA) to ensure its safety.**\n\n**The quality of water can be affected by various factors, including natural contaminants such as minerals and microorganisms, as well as man-made pollutants such as chemicals and industrial waste. The presence of harmful substances in water can cause health problems such as gastrointestinal illness, reproductive problems, and even cancer.**\n\n**To ensure water potability, water treatment processes are used to remove contaminants and ensure that the water is safe for consumption. These processes may include filtration, disinfection, and the addition of chemicals to remove impurities.**\n\n"

    st.markdown("# Water Potability💧")
    st.write("**Water potability refers to the suitability of water for consumption by humans without causing any adverse health effects. Potable water must meet certain standards and guidelines established by regulatory bodies such as the World Health Organization (WHO) and the Environmental Protection Agency (EPA) to ensure its safety.**\n\n**The quality of water can be affected by various factors, including natural contaminants such as minerals and microorganisms, as well as man-made pollutants such as chemicals and industrial waste. The presence of harmful substances in water can cause health problems such as gastrointestinal illness, reproductive problems, and even cancer.**\n\n**To ensure water potability, water treatment processes are used to remove contaminants and ensure that the water is safe for consumption. These processes may include filtration, disinfection, and the addition of chemicals to remove impurities.**\n\n")
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
        
        
    plt.title("Comare k value in model")
    plt.plot(k_neighbors, test_score, 'ro', label="Test score")
    plt.plot(k_neighbors, train_score, 'bo', label="Train score")
    plt.legend()
    plt.xlabel("K number")
    plt.ylabel("Score")
    plt.show()
if select_box == "water potability":
    st.title("# Water Potability💧")
    wt.water()
if select_box == "Prediction Prediction" :
    st.title(select_box)
    up.population()
if select_box == "about web" :
    st.title(select_box)
    st.markdown("เว็ปนี้จัดทำขึ้นเพื่อศีกษาและหาข้อมูล streamlit ")
    st.write("จัดทำโดย นายอัษฎาวุธ  ประสารคำ 65114540710")
