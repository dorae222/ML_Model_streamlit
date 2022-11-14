# 필요한 라이브러리 import
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_option_menu import option_menu
import plotly.express as px
import seaborn as sns

#import plotly.express as px
##################################################################################
# st.markdown을 통해 전체 틀 고정
st.markdown(
        f"""
<style>
    .reportview-container .main .block-container{{
        max-width: 90%;
        padding-top: 5rem;
        padding-right: 5rem;
        padding-left: 5rem;
        padding-bottom: 5rem;
    }}
    img{{
    	max-width:40%;
    	margin-bottom:40px;
    }}
</style>
""",
        unsafe_allow_html=True,
    )
##################################################################################
with st.sidebar:
    choose = option_menu("Contents", ["About","DataFrame", "Visualizing", "Predicting"],
                         icons=['house','clipboard data', 'kanban', 'person lines fill'],
                         menu_icon="app-indicator", default_index=0,
                         styles={
        "container": {"padding": "4!important", "background-color": "#fafafa"},
        "icon": {"color": "black", "font-size": "25px"}, 
        "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "#fafafa"},
        "nav-link-selected": {"background-color": "#02ab21"},
    }
    )
##################################################################################
# 파트별 컨테이너화
# 굳이 안해도 되지만, 코드 가독성이나 구조를 위해서 사이드바를 만들기 이전에 구성하였습니다.
header_container = st.container()
visualizing_container = st.container()
stats_container = st.container()
forcasting_container = st.container()
##################################################################################
# About 페이지
if choose == "About":
    with header_container:
        st.header("타이타닉 생존자 예측하기")
        st.image('img/titanic.png')
        st.subheader("Streamlit을 활용하여 ML모델을 웹으로 표현해보자!")
        st.write("처음 만들어 보는 ML 웹이지만, 즐겁게 봐주세요!")
        st.write("---")
        st.subheader("컴퓨터 세팅을 Light로 세팅해야, 목차가 제대로 보입니다.")
        st.subheader("Contents")
        st.write("흑백 모드일 시, 사이드바가 잘 보이지 않습니다.")
        st.write("사이드 바가 보이지 않는다면, 왼쪽 화살표를 클릭해주세요.")
        st.write("")
        st.write("1.DataFrame: 데이터셋을 자유자재로 살펴보세요.")
        st.write("2.Visualizing: 데이터 상관관계를 그래프로 확인해보세요.")
        st.write("3.Predicting: 변수를 조정하여 생존자를 예측해보세요.")
##################################################################################
# Visualizing 페이지
elif choose == "Visualizing":
    with visualizing_container:
        st.title("Visualizing")
        data = pd.read_csv('titanic_data.csv')

        def pre_processing(df : pd.DataFrame):
            df.Embarked = df.Embarked.fillna("S")
            df.Fare = df.Fare.fillna(0)
            df['Title'] = df.Name.str.extract('([A-Za-z]+)\.')
            rarelist = [a for a in set(df['Title'])
                        if list(df['Title']).count(a) < 10]
            df['Title'] = df['Title'].replace(rarelist, 'Rare')
            title_age_mean = df.groupby(['Title'])['Age'].mean()
            for v in df['Title'].unique():
                df.loc[df.Age.isnull() & (df.Title == v), 'Age'] = title_age_mean[v]
            df_clean = df.drop(columns=['Name', 'Ticket', 'Title', 'Cabin'])
            return pd.get_dummies(df_clean,
                                columns = ['Sex', 'Embarked'], drop_first=True)
        
        data = pre_processing(data)

        st.subheader("Plotly를 이용한 Heatmap")
        fig = px.imshow(data.corr(),text_auto=True, color_continuous_scale='RdBu_r', aspect='auto')
        st.plotly_chart(fig)
        st.write("---")
        st.subheader("Plotly를 이용한 ScatterPlot")
        fig2 = px.scatter_matrix(data,dimensions=data.columns,color="Survived")
        st.plotly_chart(fig2)
##################################################################################
# DataFrame 페이지
elif choose == "DataFrame":
    with stats_container:
        st.title("DataFrame")
        data = pd.read_csv('titanic_data.csv')

        def pre_processing(df : pd.DataFrame):
            df.Embarked = df.Embarked.fillna("S")
            df.Fare = df.Fare.fillna(0)
            df['Title'] = df.Name.str.extract('([A-Za-z]+)\.')
            rarelist = [a for a in set(df['Title'])
                        if list(df['Title']).count(a) < 10]
            df['Title'] = df['Title'].replace(rarelist, 'Rare')
            title_age_mean = df.groupby(['Title'])['Age'].mean()
            for v in df['Title'].unique():
                df.loc[df.Age.isnull() & (df.Title == v), 'Age'] = title_age_mean[v]
            df_clean = df.drop(columns=['Name', 'Ticket', 'Title', 'Cabin'])
            return pd.get_dummies(df_clean,
                                columns = ['Sex', 'Embarked'], drop_first=True)
        
        data = pre_processing(data)
        #######################################
        Pclass_list = ['All'] + data['Pclass'].unique().tolist()
        s_Pclass = st.selectbox('Pclass를 기준으로 데이터셋을 살펴보세요!', Pclass_list, key='start_station')
        #######################################
        # display the collected input
        st.write('당신은 Pclass: ' + str(s_Pclass)+'를 선택하였습니다' )
        #######################################
        if s_Pclass != 'All':
            display_data = data[data['Pclass'] == s_Pclass]

        else:
            display_data = data.copy()

        st.write(display_data)
        #######################################
        st.write('---')
        st.subheader("컬럼 정보")
        st.write("Survived: 생존 여부 => 0 = No, 1 = Yes")
        st.write("pclass: 티켓 등급 => 1 = 1st, 2 = 2nd, 3 = 3rd")
        st.write("Sex: 성별")
        st.write("Age: 나이")
        st.write("Parch: 함께 탑승한 부모, 자식의 수")
        st.write("Ticket: 티켓 번호")
        st.write("Fare: 운임")
        st.write("Embarked: 탑승 항구 => C = Cherbourg, Q = Queenstown, S = Southampton")
        #######################################
        st.write('---')
        st.subheader("기초 통계")
        st.write(data.describe())
        #######################################
        st.write("---")
##################################################################################
# Predicting 페이지
elif choose == "Predicting":
    with forcasting_container:
        st.title("Predicting")
        st.subheader("변수들을 조정하고, 예측버튼을 클릭해주세요!")
        st.subheader("변수 설명과 유의점은 아래를 참고 부탁드립니다!!")
        #######################################
        # 첫번째 행
        r1_col1, r1_col2, r1_col3 = st.columns(3)

        Pclass_option = (1,2,3)
        Pclass = r1_col1.selectbox("Pclass", Pclass_option)

        Age = r1_col2.slider("Age", 0, 80)

        SibSp = r1_col3.slider("SibSp", 0,8)
        #######################################
        # 두번째 행
        r2_col1, r2_col2= st.columns(2)

        Parch = r2_col1.slider("Parch", 0, 6)
        Fare = r2_col2.slider("Fare", 0, 247)
        #######################################
        # 세번째 행
        r3_col1, r3_col2, r3_col3= st.columns(3)

        Sex_male_option = (0,1)
        Sex_male = r3_col1.selectbox("Sex_male", Sex_male_option)

        Embarked_Q_option = (0,1)
        Embarked_Q = r3_col2.selectbox("Embarked_Q", Embarked_Q_option)

        Embarked_S_option = (0,1)
        Embarked_S = r3_col3.selectbox("Embarked_S", Embarked_S_option)
        #######################################
        # 네번째 행
        st.write("---")
        st.image('img/model_reference.png')
        r4_col1, r4_col2= st.columns(2)

        model_option = ('KNN','DecisionTree','bagging','ExtraTree','RandomForest','AdaBoosting','ExtremeBoosting','GradientBoosing','vote_soft','grid_soft')
        model_select = r4_col1.selectbox("model_option", model_option)
        
        r4_col2.write(model_select+'모델 선택하였습니다')
        # 예측 버튼
        predict_button = st.button("예측")
        #######################################
        # 예측 결과
        variable = np.array([Pclass, Age, SibSp, Parch, Fare, Sex_male, Embarked_Q,Embarked_S])

        if predict_button:
            if model_select == 'KNN':
                model = joblib.load('pkl/knn_model.pkl')
                pred = model.predict([variable])

            if model_select == 'DecisionTree':
                model = joblib.load('pkl/DecisionTree.pkl')
                pred = model.predict([variable])

            if model_select == 'bagging':
                model = joblib.load('pkl/bagging.pkl')
                pred = model.predict([variable])

            if model_select == 'ExtraTree':
                model = joblib.load('pkl/ExtraTree.pkl')
                pred = model.predict([variable])

            if model_select == 'RandomForest':
                model = joblib.load('pkl/RandomForest.pkl')
                pred = model.predict([variable])

            if model_select == 'AdaBoosting':
                model = joblib.load('pkl/AdaBoosting.pkl')
                pred = model.predict([variable])

            if model_select == 'ExtremeBoosting':
                model = joblib.load('pkl/ExtremeBoosting.pkl')
                pred = model.predict([variable])

            if model_select == 'GradientBoosing':
                model = joblib.load('pkl/GradientBoosing.pkl')
                pred = model.predict([variable])

            if model_select == 'vote_soft':
                model = joblib.load('pkl/vote_soft.pkl')
                pred = model.predict([variable])

            if model_select == 'grid_soft':
                model = joblib.load('pkl/grid_soft.pkl')
                pred = model.predict([variable])
            else:
                pass

            variable = np.array([Pclass, Age, SibSp, Parch, Fare, Sex_male, Embarked_Q,Embarked_S])
            pred = model.predict([variable])

            st.subheader("생존시:1, 사망시:0")
            st.metric("결과: ", pred[0])
        ################
        st.write("---")
        st.subheader("컬럼 정보")
        st.write("Survived: 생존 여부 => 0 = No, 1 = Yes")
        st.write("pclass: 티켓 등급 => 1 = 1st, 2 = 2nd, 3 = 3rd")
        st.write("Sex: 성별")
        st.write("Age: 나이")
        st.write("Parch: 함께 탑승한 부모, 자식의 수")
        st.write("Ticket: 티켓 번호")
        st.write("Fare: 운임")
        st.write("Embarked: 탑승 항구 => C = Cherbourg, Q = Queenstown, S = Southampton")
        st.write("---")
        st.subheader("변수 선택시 주의 사항")
        st.write("Embarked Q:1 S:1 불가능")
        st.write("Embarked Q:1 S:0 가능 => Q인 케이스")
        st.write("Embarked Q:0 S:1 가능 => S인 케이스")
        st.write("Embarked Q:0 S:0 가능 => C인 케이스")
        st.write("Sex_male: 1  남성")
        st.write("Sex_male: 0  여성")
        st.write("---")
        #######################################
