![header](https://capsule-render.vercel.app/api?type=waving&color=auto&height=200&section=header&text=ML%20MODEL%20with%20Streamlit&fontSize=50&animation=fadeIn&fontAlignY=30&desc=Using%20ML%20models%20in%20Web%20pages%20with%20TitanicData&descAlignY=51&descAlign=62)

# 다양한 머신러닝 모델을 이용한 타이타닉 생존자 예측 웹 구축
### Link: https://dorae222-knn-appyling-streamlit-s-knn-algorithm-gixk3h.streamlit.app/

#### <데이터셋: 타이타닉 데이터><br>
  - 출처: https://www.kaggle.com/c/titanic<br>
  - 모델링 참고: https://www.kaggle.com/code/ldfreeman3/a-data-science-framework-to-achieve-99-accuracy<br>
  - 데이터 셋 선택이유: 모델링 보다는 처음 써보는 streamlit을 활용한 모델을 웹으로 구현하는 부분에 초점을 두고 싶었음.<br>
    - 기본 선형회귀부터  Random Forest(Bagging 방식), Gradient Boosting Machines(Boosting 기법) 등 다양한 모델을 적용하였음.
#### <예시 페이지>
![image](https://user-images.githubusercontent.com/105966480/201268970-823d6ba9-46ca-499d-8f1d-17ada1157ec7.png) |![image](https://user-images.githubusercontent.com/105966480/201268931-e7c38c3a-446b-438b-92d4-5ba563d0ef71.png)
--- | --- | 
![image](https://user-images.githubusercontent.com/105966480/201742474-d3fd850e-fda7-4c52-913a-bbeb525e3612.png) |![image](https://user-images.githubusercontent.com/105966480/201743245-0abbeaaa-495e-4445-94c5-9f9fc19200d1.png)
#### <개선사항>
  1. 그냥 변수를 보면 무슨 내용인지 이해하기 어려움.<br>
    - 그래서 데이터 프레임을 자유롭게 다룰 수 있게 설정하였다.
  2. st.container()를 이용하여, 파트를 명확하게 구분
  3. st.markdown을 활용하여, 전체 틀 고정
  4. 사이드바를 추가하여, 파트 분리
  5. Plotly를 활용한 그래프 삽입
  6. 데이터와 웹의 연결은 아니지만 모델을 형성하기 위해 전처리와 파라미터를 수정<br>
    - https://github.com/dorae222/knn_appyling_streamlit/blob/main/Modeling.ipynb
#### <작업시 겪었던 난항>
1. 스케일러도 pkl로 저장해서 새 기준에 대해 transform하려 했지만, 굳이 할 필요가 없었다.
2. 바보 같지만 .txt를 .text로 저장해서 계속 난항을 겼었는데, 파일명에 주의하자!<br>
  - requirements.txt 파일에 추가로 설치해야되는 라이브러리를 언급해줘야 한다.
    - 예를 들어, from skitlearn.preprocessing~ => scikit-learn을 적어줘야 한다.<br>
    ![image](https://user-images.githubusercontent.com/105966480/201176417-b04b1385-6ce4-4f18-a488-47d4d591d996.png)<br>
  - 확장자명을 반드시 지켜야 하며, py는 이름이 변해도 상관없지만 requiremets.txt는 고정이다.<br>
    ![image](https://user-images.githubusercontent.com/105966480/201176243-6408100b-0472-4e5c-87c9-4bca5404004a.png)
3. 데이터 변환 과정에서 차원이 어떻게 변하는지 머리속으로 잘 그리자.<br>
  - ValueError: Expected 2D array, got 1D array instead:<br>
  ![image](https://user-images.githubusercontent.com/105966480/201177873-bab43a09-ef37-4670-bbdf-689ca8c991af.png)<br>
4. 모델링을 우리 데이터셋에 맞게 수정하고, 웹으로 구현하는게 쉽지 않음을 느꼈다.<br>
  - 다양한 모델들을 어떻게 보여줄 것인지가 가장 고민이 되었다.<br>
  ![image](https://user-images.githubusercontent.com/105966480/201745894-f1d574b1-613e-4cda-867b-5399d76a377a.png)
