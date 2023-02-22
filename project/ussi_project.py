import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib
import time as t
def page():
    def population():
        st.markdown(
            f"""
               <style>
               .stApp {{
                   background-image: url("https://wallpapercave.com/wp/wp4890462.jpg");
                   background-attachment: fixed;
                   background-size: cover;
                   /* opacity: 0.3; */
               }}
               </style>
               """,
            unsafe_allow_html=True
        )
        st.title('Population Prediction')
        # st.sidebar.title('Menu')
        # st.sidebar.selectbox("Were are you go",('Prediction Prediction','Noting Now'))
        left, right = st.columns(2)
        left.markdown("เว็ปคาดคะเนจำนวนประชากร")
        # left.markdown("โดยที่ผู้ใช้กรอกขนาดพื้นที่ของบ้าน หน่วย ตารางวา")

        def generate_data():
            t0 = int(t.time())
            data=pd.read_csv("./project/dataset.csv")
            data=pd.DataFrame(data)
            x=data["B.E"]
            y=data['sum']
            df = pd.DataFrame({
                'x': x,
                'y': y
            })
            with st.spinner() :
                t1 = int(t.time())
                t.sleep(1 + t1 - t0)
            df.to_excel('./project/data.xlsx')

        def load_data():
            return pd.read_excel('./project/data.xlsx')

        def save_model(model):
            joblib.dump(model, './project/linear_regression.joblib')

        def load_model():
            return joblib.load('./project/linear_regression.joblib')

        generateb = right.button('generate data.xlsx')
        if generateb:
            right.write('generating "data.xlsx" ...')
            generate_data()
            st.spinner()
            right.success("... done")

        loadb = right.button('load data.xlsx')
        if loadb:
            t0 = int(t.time())
            df = pd.read_excel("./project/data.xlsx",header=0, names=['B.E', 'sum'] ,index_col=None)
            right.write('loading "data.xlsx ..."')
            with st.spinner() :
                t1 = int(t.time())
                t.sleep(1 + t1 - t0)
            right.success('... done')
            right.dataframe(df)
            fig, ax = plt.subplots()
            df.plot.scatter(x='B.E', y='sum', ax=ax)
            st.pyplot(fig)

        trainb = right.button('train model')
        if trainb:
            t0 = int(t.time())
            data = load_data()
            d = pd.DataFrame(data)
            # x = d.drop(columns="x",axis=0)
            # x=np.array(x).astype(np.int16)
            # y = d['x']
            # y=np.array(y).astype(np.int32)
            right.write('training model ...')
            # x_train,x_test,y_train,y_test=train_test_split(x,y,test_size= 0.8)
            model = LinearRegression()
            model.fit(data.x.values.reshape(-1,1),data.y)
            with st.spinner() :
                t1 = int(t.time())
                t.sleep(1 + t1 - t0)
            right.success('... done')
            right.dataframe(d)
            save_model(model)
        BC = left.number_input('ปี(พ.ศ.)',step=1)
        predictb = left.button('คาดคะเน')
        if predictb:
            model = load_model()
            # m=joblib.load('./project/linear_regression.joblib')
            predict = model.predict(np.array([int(BC)]).reshape(-1,1))
            left.markdown(f'จำนวนประชาการในปี :green[{int(BC)}] ประมาณ :red[{int(predict[0])} คน]')