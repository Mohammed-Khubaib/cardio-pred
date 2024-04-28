# Imports
import streamlit as st
from st_on_hover_tabs import on_hover_tabs
import json
from streamlit_lottie import st_lottie
import numpy as np
import pandas as pd 
# import matplotlib.pyplot as plt
from data_desciption import describe
from NoiseDetection import boxplots,NoiseRemoved
from diagrams import fig
from dataprocessing import dataprocessing
# ML related Imports:
from sklearn.preprocessing import MinMaxScaler
import joblib
import os
# Disable Arrow serialization
os.environ["PANDAS_ARROW_ALWAYS_VERIFY"] = "True"

# Page Initalization
st.set_page_config(page_title="MediPredict", page_icon='🏥', layout="wide", initial_sidebar_state='expanded')
# Hide the "Made with Streamlit" footer
hide_streamlit_style = """
    <style>
    #MainMenu{visibility:hidden;}
    footer{visibility:hidden;}
    </style>
    hr {
            color: #803df5 ;
        }
"""
# st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# Loading the Lottie files for animations
def load_lottie_file(filepath:str):
    with open(filepath,"r") as f:
        return json.load(f)
lottie_file1 =load_lottie_file('./HeartAnimation.json')
lottie_file2 =load_lottie_file('./HeartPulse.json')

# Defining the sidebar
with st.sidebar:
    st_lottie(lottie_file2,speed=0.8,reverse=False,height=100,width=300)
    # tabs = on_hover_tabs(tabName=['Dashboard','Data Pre Processing','K-Nearest Neighbor','Support Vector Machine'], 
    tabs = on_hover_tabs(tabName=['Dashboard','Data Pre Processing','K-Nearest Neighbor','Support Vector Machine','Decision Tree','Random Forest'], 
                         iconName=['web_asset','bar_chart_4_bars','scatter_plot','blur_linear','lan','forest'], default_choice=0,
                         styles={'navtab': {'background-color':'#dde6ed',
                                            'color': '#1593af',
                                            'font-size': '18px',
                                            'transition': '.3s',
                                            'white-space': 'nowrap',
                                            'text-transform': 'uppercase'},
                                 'tabOptionsStyle': {':hover :hover': {'color': '#004280',
                                                                     'cursor': 'pointer'}},
                             },
    )

# Loading The dataset
Main_Dataset = pd.read_csv('./data/heart.csv')
if tabs == 'Dashboard':
    c1 , c2 = st.columns([0.3,0.7])
    with c1:
        st_lottie(lottie_file1,speed=0.5,reverse=False,height=350,width=350)
    with c2:
        st.title(':blue[Cardio Predict]',anchor=False)
        st.markdown("""

        **Heart Healty Condition Predictor App** 
        """)
        st.markdown("""
        - is essential for early detection of cardiovascular risks, aiding in timely interventions and lifestyle adjustments to prevent heart-related ailments and reduce healthcare burdens.
        - It can lead to early interventions, better management of health conditions, and improved patient outcomes. 
        - To this end, we propose the development of a Cardio Predict Model using Machine Learning (ML) techniques.

        This model will analyze various health parameters of an individual and predict the likelihood of them developing a Heart disease.

        _The parameters could include_ `age	,sex ,cp ,trtbps ,chol,fbs,restecg,thalachh	,exng,oldpeak,slp,caa,thall`, among others._""")
    code = """
        <hr style="border-top: 3px solid #1593af; border-bottom: none;">

    """
    st.html(code)
    status = st.toggle("Load Model")
    # st.warning(f"statis = {status}")
    if status:
        loaded_model = joblib.load('svm_model.pkl')
        cx,cy = st.columns([0.5,0.5])
        with cx:
            Age = st.number_input("1. Enter the Age",min_value=16,max_value=77,step=1,value=67)
            Sex= st.radio("2. Male or Female",['M','F'],horizontal=True)
            Sex = 0 if Sex == 'F' else 1
            ChestPainType = st.radio('3. Chest Pain Type', ('Typical angina' ,'Atypical angina' ,'Non-anginal pain' ,'Asymptomatic'),horizontal=True)
            if ChestPainType == 'Typical angina':
                ChestPainType = 0
            elif ChestPainType == 'Atypical angina':
                ChestPainType = 1
            elif ChestPainType == 'Non-anginal pain':
                ChestPainType = 2
            else :
                ChestPainType = 3
            RestingECG = st.radio('4. Resting Electrocardiogram', ('Normal', 'ST', 'LVH'),horizontal=True)
            if RestingECG == 'Normal':
                RestingECG = 0
            elif RestingECG == 'ST':
                RestingECG = 1
            else:
                RestingECG = 2
            ExerciseAngina = st.radio('ExerciseAngina', ('Yes', 'No'),horizontal=True,index=0)
            ExerciseAngina = 0 if ExerciseAngina == 'No' else 1
            ST_Slope = st.radio('5. ST Slope', ('Up', 'Flat', 'Down'),horizontal=True,index=1)
            if ST_Slope == 'Up':
                ST_Slope = 0
            elif ST_Slope == 'Flat':
                ST_Slope = 1
            else:
                ST_Slope = 2
            FastingBS = st.radio('6. Fasting Blood Sugar', (0, 1),horizontal=True)
            
        with cy:
            RestingBP = st.slider('7. Resting Blood Pressure', 94, 200,value=160)
            Cholesterol = st.slider('8. Cholesterol',  126, 603,value=286)
            MaxHR = st.slider('9. Maximum Heart Rate', 71, 202,value=108)
            Oldpeak = st.slider('10. Old peak', min_value=0.2, max_value=6.2,step=0.1,value=1.5)
            Caa = st.slider("11. number of major vessels (0-4)",min_value=0,max_value=4,step=1,value=3)
            ThallHelp = """
                0: Normal blood flow observed.\n
                1: Mildly abnormal blood flow observed.\n
                2: Moderately abnormal blood flow observed.\n
                3: Severely abnormal blood flow observed.
            """
            Thall = st.slider("12. Thallium Stress Test Result (0-4)",min_value=0,max_value=4,step=1,help=ThallHelp,value=2)
        data = {
            'age': Age,
            'sex': Sex,
            'cp': ChestPainType,
            'trtbps': RestingBP,
            'chol': Cholesterol,
            'fbs': FastingBS,
            'restecg': RestingECG,
            'thalachh': MaxHR,
            'exng': ExerciseAngina,
            'oldpeak': Oldpeak,
            'slp': ST_Slope,
            'caa': Caa,
            'thall': Thall
        }
        features = pd.DataFrame(data, index=[0])
        st.dataframe(features,use_container_width=True)
        model_var_param = {
            'age': {'mean': 54.366337, 'std': 9.082101, 'min': 29, 'max': 77}, 
            'sex': {'mean': 0.683168, 'std': 0.466011, 'min': 0, 'max': 1}, 
            'cp': {'mean': 0.966997, 'std': 1.032052, 'min': 0, 'max': 3}, 
            'trtbps': {'mean': 131.623762, 'std': 17.538143, 'min': 94.0, 'max': 200}, 
            'chol': {'mean': 246.264026, 'std': 51.830751, 'min': 126.0, 'max': 564}, 
            'fbs': {'mean': 0.148515, 'std': 0.356198, 'min': 0, 'max': 1}, 
            'restecg': {'mean': 0.528053, 'std': 0.52586, 'min': 0, 'max': 2}, 
            'thalachh': {'mean': 149.646865, 'std': 22.905161, 'min': 71.0, 'max': 202}, 
            'exng': {'mean': 0.326733, 'std': 0.469794, 'min': 0, 'max': 1}, 
            'oldpeak': {'mean': 1.039604, 'std': 1.161075, 'min': 0, 'max': 6.2}, 
            'slp': {'mean': 1.39934, 'std': 0.616226, 'min': 0, 'max': 2}, 
            'caa': {'mean': 0.729373, 'std': 1.022606, 'min': 0, 'max': 4}, 
            'thall': {'mean': 2.313531, 'std': 0.612277, 'min': 0, 'max': 3}
        }
        for k in model_var_param.keys():
            # st.warning(k)
            min = model_var_param[k]['min']
            # st.warning(min)
            max = model_var_param[k]['max']
            # st.warning(max)
            # st.warning(f"{k} = {data[k]}")
            min_max = (data[k] - min)/(max-min)
            # st.warning(f"scaled {k} = {min_max}")
            data[k] = min_max
        patent_data = pd.DataFrame(data, index=[0])
        # st.dataframe(patent_data,use_container_width=True)
        if st.button("Predict Result"):
            y_pred_loaded = loaded_model.predict(patent_data)
            cp,cq,cr=st.columns([0.4,0.7,0.2])
            with cq:
                if y_pred_loaded[0]==0:
                    lottie_file3 =load_lottie_file('./GreenHeartPulse.json')
                    with st.container():
                        st_lottie(lottie_file3,speed=0.5,reverse=False,height=250,width=550)
                        st.balloons()
                        st.title(":green[Patients Heart is Healthy]",anchor=False)
                if y_pred_loaded[0]==1:
                    with st.container():
                        st_lottie(lottie_file2,speed=0.5,reverse=False,height=250,width=550)
                        st.title(":red[Patients Heart is Not Healthy]",anchor=False)

if tabs =='Data Pre Processing':
    c1 , c2 = st.columns([0.3,0.7])
    with c1:
        st_lottie(lottie_file1,speed=0.5,reverse=False,height=150,width=150)
    with c2:
        st.title(':blue[Cardio Predict:]',anchor=False)
    listTabs = [
        '$$\Large\\textbf{Dataset}$$',
        '$$\Large\\textbf{Data Descriptions}$$',
        '$$\Large\\textbf{Noise Detection}$$',
        '$$\Large\\textbf{Comparision}$$',
        '$$\Large\\textbf{Normalize:}$$',
        ]
    whitespace = 2
    tab_labels = [s.center(len(s) + whitespace, "\u2001") for s in listTabs]
    sections = st.tabs(tab_labels)
    # Main_Dataset = pd.read_csv('./data/heart.csv')
    # tabs = Tabs()
    with sections[0]:
        st.header(':blue[Dataset :]',anchor=False)
        st.dataframe(Main_Dataset,use_container_width=True,hide_index=True)
    with sections[1]:
        st.header(':blue[Data Description :]',anchor=False)
        output = describe(Main_Dataset)
        output_df = pd.DataFrame(output)
        st.dataframe(output_df,use_container_width=True,hide_index=True)
    with sections[2]:
        st.header(':blue[Noise Detection :]',anchor=False)
        # boxplots(Main_Dataset)
        st.image("./data/plots/n1.png")
        st.image("./data/plots/n2.png")
        st.image("./data/plots/n3.png")
        st.image("./data/plots/n4.png")
        st.image("./data/plots/n5.png")
        # Now Removing the Outlier that was spotted in chol > 500
        st.info("Cholesterol level greater than 500 is an outlier, so we are removing those data points from the dataset.")
        Chol_noise = Main_Dataset[Main_Dataset["chol"]>500].index
        Main_Dataset.drop(index=[Chol_noise[0]], inplace=True)
        # st.warning(Main_Dataset.shape)
        # NoiseRemoved(Main_Dataset)
        st.image("./data/plots/n6.png")
        
    with sections[3]:
        st.header(":blue[Comparisions :]",anchor=False)
        plots = fig(Main_Dataset)
        st.subheader(":blue[Chest Pain Type]:red[ vs] :blue[Age] :",anchor=False)
        # plots.age_cp()
        st.image('./data/plots/AgeCp.png')
        st.subheader(":blue[Fasting blood sugar]:red[ vs] :blue[Age] :",anchor=False)
        # plots.age_fbs()
        st.image('./data/plots/AgeFbs.png')
        st.subheader(":blue[Resting Electrocardiographic results]:red[ vs] :blue[Age] :",anchor=False)
        # plots.age_rectecg()
        st.image('./data/plots/AgeRestecg.png')
        st.subheader(":blue[exercise induced angina]:red[ vs] :blue[Age] :",anchor=False)
        # plots.age_exng()
        st.image('./data/plots/AgeExng.png')
        st.subheader(":blue[ST Segment Slope]:red[ vs] :blue[Age] :",anchor=False)
        # plots.age_slp()
        st.image('./data/plots/AgeSLP.png')
        st.subheader(":blue[number of major vessels]:red[ vs] :blue[Age] :",anchor=False)
        # plots.age_caa()
        st.image('./data/plots/AgeCaa.png')
        st.subheader(":blue[maximum heart rate achieved]:red[ vs] :blue[Age] :",anchor=False)
        # plots.age_thall()
        st.image('./data/plots/AgeThall.png')
    with sections[4]:
        # normalize the entire dataset (except target)
        st.header(":blue[Normalised Dataset]",anchor=False)
        Features = Main_Dataset.drop(columns='output')
        Features = pd.DataFrame(Features)
        new_output = describe(Features)
        # st.write(new_output)
        output_df = pd.DataFrame(new_output)
        st.dataframe(output_df,use_container_width=True,hide_index=True)
        # print(Features.dtypes)
        Features = pd.DataFrame(Features)
        Fcols = Features.columns
        Features[Fcols] = Features[Fcols].astype(float)
        # print(Features.dtypes)
        scaler = MinMaxScaler()
        Norm_data = scaler.fit_transform(Features)
        Norm_df = pd.DataFrame(Norm_data, columns= Features.columns)
        st.dataframe(Norm_df,use_container_width=True,hide_index=True)
        # new_output = describe(Norm_df)
        # st.dataframe(new_output,use_container_width=True,hide_index=True)

if tabs == 'K-Nearest Neighbor':
    c1 , c2 = st.columns([0.3,0.7])
    with c1:
        st_lottie(lottie_file1,speed=0.5,reverse=False,height=150,width=150)
    with c2:
        st.title(':blue[Cardio Predict]',anchor=False)
    st.header('K-Nearest Neighbor :',anchor=False)
    st.image('./data/plots/KnnPlot1.png')
    # KNN Tabs:
    listTabs = [
        '$$\Large\\textbf{Hyperparameter P = 1}$$',
        '$$\Large\\textbf{Hyperparameter P = 2}$$',
    ]
    whitespace = 5
    tab_labels = [s.center(len(s) + whitespace, "\u2001") for s in listTabs]
    hyperparameter_tabs = st.tabs(tab_labels)
    with hyperparameter_tabs[0]:
        st.subheader("Number of Neighbor (K) = 3 , Hyper parameters (P) = 1")
        st.success(f"Accuracy : 0.902")
        Best_knn =  0.902
        st.image('./data/plots/knn1.png')
    with hyperparameter_tabs[1]:
        st.subheader("Number of Neighbor (K) = 3 , Hyper parameters (P) = 2")
        st.warning(f"Accuracy : 0.869")
        st.image('./data/plots/knn2.png')


if tabs == 'Support Vector Machine':
    c1 , c2 = st.columns([0.3,0.7])
    with c1:
        st_lottie(lottie_file1,speed=0.5,reverse=False,height=150,width=150)
    with c2:
        st.title(':blue[Cardio Predict:]',anchor=False)
    st.header('Support Vector Machine :',anchor=False)
    # SVM Tabs:
    listTabs = [
        '$$\Large \\textbf{Linear}$$',
        '$$\Large\\textbf{Poly}$$',
        '$$\Large\\textbf{rbf}$$',
        '$$\Large\\textbf{sigmoid}$$',
    ]
    whitespace = 5
    tab_labels = [s.center(len(s) + whitespace, "\u2001") for s in listTabs]
    hyperparameter_tabs = st.tabs(tab_labels)
    with hyperparameter_tabs[0]:
        st.image('./data/plots/svm1.png')
        st.success(f"Accuracy :  0.934")
        Best_SVM = 0.934
        st.image('./data/plots/svmL.png')
    with hyperparameter_tabs[1]:
        st.image('./data/plots/svm2.png')
        st.warning(f"Accuracy : 0.885")
        st.image('./data/plots/svmP.png')
    with hyperparameter_tabs[2]:
        st.image('./data/plots/svm3.png')
        st.info(f"Accuracy : 0.902")
        st.image('./data/plots/svmR.png')
    with hyperparameter_tabs[3]:
        st.image('./data/plots/svm4.png')
        st.warning(f"Accuracy : 0.885")
        st.image('./data/plots/svmS.png')

if tabs == 'Decision Tree':
    c1 , c2 = st.columns([0.3,0.7])
    with c1:
        st_lottie(lottie_file1,speed=0.5,reverse=False,height=150,width=150)
    with c2:
        st.title(':blue[Cardio Predict:]',anchor=False)
    st.header('Decision Tree :',anchor=False)
    # DT Tabs:
    listTabs = [
        '$$\Large\\textbf{Gini}$$',
        '$$\Large\\textbf{Entropy}$$',
        '$$\Large\\textbf{Log Loss}$$',
    ]
    whitespace = 5
    tab_labels = [s.center(len(s) + whitespace, "\u2001") for s in listTabs]
    hyperparameter_tabs = st.tabs(tab_labels)
    with hyperparameter_tabs[0]:
        st.image('./data/plots/dt1.png')
        st.success(f"Accuracy :  0.819")
        st.image('./data/plots/dtc1.png')
    with hyperparameter_tabs[1]:
        st.image('./data/plots/dt2.png')
        st.warning(f"Accuracy :  0.868")
        Best_SVM = 0.868
        st.image('./data/plots/dtc2.png')
    with hyperparameter_tabs[2]:
        st.image('./data/plots/dt3.png')
        st.warning(f"Accuracy :  0.868")
        st.image('./data/plots/dtc3.png')

    


if tabs == 'Random Forest':
    c1 , c2 = st.columns([0.3,0.7])
    with c1:
        st_lottie(lottie_file1,speed=0.5,reverse=False,height=150,width=150)
    with c2:
        st.title(':blue[Cardio Predict:]',anchor=False)
    st.header('Random Forest :',anchor=False)
    # RF Tabs:
    listTabs = [
        '$$\Large\\textbf{Gini}$$',
        '$$\Large\\textbf{Entropy}$$',
    ]
    whitespace = 5
    tab_labels = [s.center(len(s) + whitespace, "\u2001") for s in listTabs]
    hyperparameter_tabs = st.tabs(tab_labels)
    with hyperparameter_tabs[0]:
        st.image('./data/plots/RF1.png')
        st.success(f"Accuracy :  0.852")
        st.image('./data/plots/rfc1.png')
    with hyperparameter_tabs[1]:
        st.image('./data/plots/RF2.png')
        st.warning(f"Accuracy :  0.852")
        Best_SVM = 0.852
        st.image('./data/plots/rfc2.png')