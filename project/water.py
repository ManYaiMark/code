import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import time as t
import random as rd
#http://m.gettywallpapers.com/wp-content/uploads/2021/12/Desktop-Wallpapers.jpg
st.set_option('deprecation.showPyplotGlobalUse', False)
def water():
    st.markdown(
        f"""
               <style>
               .stApp {{
                   background-image: url("http://m.gettywallpapers.com/wp-content/uploads/2021/12/Desktop-Wallpaper-2022.jpg");
                   background-attachment: fixed;
                   background-size: cover;
                   /* opacity: 0.3; */
               }}
               </style>
               """,
        unsafe_allow_html=True
    )
    st.title("water potability")
    left,right = st.columns(2)
    def save_model(model):
        joblib.dump(model, './project/water_drop.joblib')
    def load_model():
        return joblib.load('./project/water_drop.joblib')
    def generate_data():
        dm_data=pd.read_csv("./project/water_drop.csv")
        dm_data=pd.DataFrame(dm_data)
        dm_data.to_excel("./project/water_drop.xlsx")
    ss =left.button("generate model")
    load=left.button("Load model")
    if load:
        load_data=pd.read_excel("./project/water_drop.xlsx")
        load_data=load_data.drop(columns="Unnamed: 0",axis=1)
        right.dataframe(load_data)

        x=load_data.drop(columns="Potability",axis=1)
        x=np.asarray(x).astype(np.float16)
        y=load_data["Potability"]
        y=np.asarray(y).astype(np.float32)
        # load_data = pd.DataFrame({
        #     'x':x,
        #     'y':y
        # })
        # load_data.plot.scatter(x=x,y=y,ax=ax)
        #
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4)

        k_neighbors = np.arange(1, 51)
        train_score = np.empty(len(k_neighbors))
        test_score = np.empty(len(k_neighbors))

        for i, k in enumerate(k_neighbors):
            knn = KNeighborsClassifier(n_neighbors=k)
            knn.fit(x_train, y_train)
            train_score[i] = knn.score(x_train, y_train)
            test_score[i] = knn.score(x_test, y_test)
            # print(test_score[i] * 100)

        plt.title("Comare k value in model")
        plt.plot(k_neighbors, test_score, 'ro', label="Test score")
        plt.plot(k_neighbors, train_score, 'bo', label="Train score")
        plt.legend()
        plt.xlabel("K number")
        plt.ylabel("Score")
        # fig, ax = plt.subplots()
        # x=plt.plot(k_neighbors, test_score, 'ro', label="Test score")
        # y=plt.plot(k_neighbors, train_score, 'bo', label="Train score")
        # x_arr = np.asarray(k_neighbors)
        # y_arr = np.asarray(test_score)
        # df = pd.DataFrame({
        #     "x": x_arr,
        #     "y": y_arr
        # })
        # df.plot.scatter(x="x", y="y", ax=ax)
        # st.pyplot(fig)
        plt.title("Comparison of test score and train score with different k values")
        plt.plot(k_neighbors, test_score, 'ro', label="Test score")
        plt.plot(k_neighbors, train_score, 'bo', label="Train score")
        plt.legend()
        plt.xlabel("Number of neighbors (k)")
        plt.ylabel("Score")
        st.pyplot()
    train=left.button("train model")
    if ss :
        generate_data()

    if train :
        dm_data = pd.read_csv("./project/water_drop.csv")
        x=dm_data.drop(["Potability"],axis=1).values
        x = np.array(x)
        y=dm_data["Potability"].values
        y = np.array(y)
        x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.4,stratify=y,random_state=2)
        model=KNeighborsClassifier()
        model.fit(x_train,y_train)
        save_model(model)

    ph=st.number_input("Input PH",min_value=0.2274,max_value=14.0000,step=.0001)
    st.markdown("min=0.2274  max=14.0000")
    Hardness=st.number_input("Input Hardness",min_value=73.4922,max_value=317.3381,step=.0001)
    st.markdown("min=73.4922  max=317.3381")
    Solids=st.number_input("Input Solids",min_value=320.9426,max_value=56488.6724,step=.0001)
    st.markdown("min=320.9426  max=56488.6724")
    Chloramines=st.number_input("Input Chloramines",min_value=1.39087,max_value=13.1270,step=.0001)
    st.markdown("min=1.39087  max=13.1270")
    Sulfate=st.number_input("Input Sulfate",min_value=129.0000,max_value=481.0306,step=.0001)
    st.markdown("min=129.0000  max=481.0306")
    Conductivity=st.number_input("Input Conductivity",min_value=201.6197,max_value=753.3426,step=.0001)
    st.markdown("min=201.6197  max=753.3426")
    Organic_carbon=st.number_input("Input Organic_carbon",max_value=27.0067,min_value=2.2000,step=.0001)
    st.markdown("min=2.2000  max=27.0067")
    Trihalomethanes=st.number_input("Input Trihalomethanes",min_value=8.5770,max_value=124.0000,step=.0001)
    st.markdown("min= 8.5770  max=124.0000")
    Turbidity=st.number_input("Input Turbidity",max_value=6.4947,min_value=1.4500,step=.0001)
    st.markdown("min=1.4500  max=6.4947")

    def pdict():
        model = load_model()
        data_key = (float(ph), float(Hardness), float(Solids), float(Chloramines), float(Sulfate), float(Conductivity), float(Organic_carbon), float(Trihalomethanes), float(Turbidity),)
        data_array = np.asarray(data_key).reshape(1, -1)
        model_data = model.predict(data_array)
        model_data=model_data
        if model_data[0] == 0:
            ts = "กินไม่ได้"
        else:
            ts = "กินได้"
        st.markdown(
            f'ถ้าค่า : ph : {ph}, Hardness : {Hardness},Solids : {Solids},Chloramines : {Chloramines},Sulfate : {Sulfate},Conductivity : {Conductivity},Organic_carbon : {Organic_carbon},Trihalomethanes : {Trihalomethanes},Turbidity : {Turbidity}')
        st.title(ts)
        left.title(ts)
    if left.button("ประเมิณคุณภาพน้ำ"):
        pdict()
    if st.button("ประเมิณคุณภาพน้ำ "):
        pdict()





