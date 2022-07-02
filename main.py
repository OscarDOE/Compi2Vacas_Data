import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot 
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


#---------------------------------#
#Global variables


#---------------------------------#



#---------------------------------#
# Page layout
## Page expands to full width
st.set_page_config(page_title='OLC2 Machine Learning App',
    layout='wide')

#---------------------------------#



#---------------------------------#
st.write("""
# Compiladores 2 - Data Science
""")

#---------------------------------#



#---------------------------------#
# Sidebar - Collects user input features into dataframeº
with st.sidebar.header('1. Upload your .CSV'):
    uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
    

#---------------------------------#




#---------------------------------#
# Displays the dataset
def lineal(datacsv):
    print(datacsv)
    
    encabezados = " "
    optionH = " "
    X = []
    
    for d in data:
        encabezados = encabezados.strip() + d +','
    encabezados = encabezados[:-1]
    encabezados = encabezados.split(',')
    encabezados.insert(0," ")

    with st.sidebar.header('3. Choose the X axis to evaluate'):
            optionA = st.sidebar.selectbox(
            'Choose X:',
            (encabezados))

    with st.sidebar.header('4. Choose the Y axis to evaluate'):
            optionH = st.sidebar.selectbox(
            'Choose Y:',
            (encabezados))


    if optionH != " " and optionA != " ":
        X = np.asarray(datacsv[optionA]).reshape(-1, 1)
        Y = data[optionH]
        linear_regression = LinearRegression()
        linear_regression.fit(X, Y)
        Y_pred = linear_regression.predict(X)

        st.markdown('**1.2. Variable details**:')

        col1,col2 = st.columns(2)
        with col1:
            st.write('X variable')
            st.write(X)
        with col2:
            st.write('Y variable')
            # Y = Y.rename({0: 'Y'})
            st.write(Y_pred)

        st.write("Error medio: ")
        st.info(mean_squared_error(Y, Y_pred, squared=True))
        st.write("Coef: ")
        st.info(linear_regression.coef_)
        st.write("R2: ")
        st.info(r2_score(Y, Y_pred))


        with st.sidebar.header(''):
            title = st.text_input('Type the prediction data')

        matplotlib.pyplot.scatter(X, Y)
        matplotlib.pyplot.plot(X, Y_pred, color='red')
        matplotlib.pyplot.savefig("graficaLineal.png")
        st.write("Graficar puntos: ")

        colu1, colu2, colu3 = st.columns([1,4,1])
        with colu1:
            st.write("")
        with colu2:
            st.image("./graficaLineal.png")
        with colu3:
            st.write("")

        coe = linear_regression.coef_[0]
        inter=linear_regression.intercept_
        coe2="{0:.4f}".format(coe)
        inter2="{0:.4f}".format(inter)
        st.write("Funcion de tendencia: ")
        if float(coe2)>0:
            st.latex(f'''
            y = {str(inter2)} + {str(coe2)} x
            ''')
        else:
            st.latex(f'''
            y = {str(inter2)} {str(coe2)} x
            ''')

        
        if title != "":
            titleInt = int(title)
            st.write("Prediction: ")
            Y_new = linear_regression.predict([[int(titleInt)]])
            st.info(Y_new)
            
        else:
            st.info('Awaiting for prediction value')

    else:
        st.info('Awaiting for X and Y axis')




def polinomial(datacsv):
    pass





if uploaded_file is not None:
    st.subheader('1. Dataset')
    data = pd.read_csv(uploaded_file)
    st.markdown('**1.1. Data set overview**')
    st.write(data)


    with st.sidebar.header('2. What do you want to do?'):
     option = st.sidebar.selectbox(
      'Choose an algorithm:',
     ('','Regresión lineal', 'Regresión polinomial', 'Clasificador Gaussiano','Clasificador de árboles de decisión','Redes neuronales'))

    if(option != ""):
        if(option == "Regresión lineal"):
            
            lineal(data)
        if(option == "Regresión polinomial"):
            polinomial(data)

        # with st.sidebar.header('5. Operations'):
        #     algoSelected = st.sidebar.selectbox(
        #     'Choose an operation:',
        #     ('','Graficar puntos', ' Definir función de tendencia', 'Realizar predicción de la tendencia','Clasificar'))
    
else:
    st.info('Awaiting for CSV file')























# import streamlit as st
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.metrics import mean_squared_error, r2_score
# from sklearn.datasets import load_diabetes, load_boston

# #---------------------------------#
# # Page layout
# ## Page expands to full width
# st.set_page_config(page_title='The Machine Learning App',
#     layout='wide')

# #---------------------------------#
# # Model building
# def build_model(df):
#     X = df.iloc[:,:-1] # Using all column except for the last column as X
#     Y = df.iloc[:,-1] # Selecting the last column as Y

#     # Data splitting
#     X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=(100-split_size)/100)
    
#     st.markdown('**1.2. Data splits**')
#     st.write('Training set')
#     st.info(X_train.shape)
#     st.write('Test set')
#     st.info(X_test.shape)

#     st.markdown('**1.3. Variable details**:')
#     st.write('X variable')
#     st.info(list(X.columns))
#     st.write('Y variable')
#     st.info(Y.name)

#     rf = RandomForestRegressor(n_estimators=parameter_n_estimators,
#         random_state=parameter_random_state,
#         max_features=parameter_max_features,
#         criterion=parameter_criterion,
#         min_samples_split=parameter_min_samples_split,
#         min_samples_leaf=parameter_min_samples_leaf,
#         bootstrap=parameter_bootstrap,
#         oob_score=parameter_oob_score,
#         n_jobs=parameter_n_jobs)
#     rf.fit(X_train, Y_train)

#     st.subheader('2. Model Performance')

#     st.markdown('**2.1. Training set**')
#     Y_pred_train = rf.predict(X_train)
#     st.write('Coefficient of determination ($R^2$):')
#     st.info( r2_score(Y_train, Y_pred_train) )

#     st.write('Error (MSE or MAE):')
#     st.info( mean_squared_error(Y_train, Y_pred_train) )

#     st.markdown('**2.2. Test set**')
#     Y_pred_test = rf.predict(X_test)
#     st.write('Coefficient of determination ($R^2$):')
#     st.info( r2_score(Y_test, Y_pred_test) )

#     st.write('Error (MSE or MAE):')
#     st.info( mean_squared_error(Y_test, Y_pred_test) )

#     st.subheader('3. Model Parameters')
#     st.write(rf.get_params())


    

# #---------------------------------#
# st.write("""
# # The Machine Learning App
# In this implementation, the *RandomForestRegressor()* function is used in this app for build a regression model using the **Random Forest** algorithm.
# Try adjusting the hyperparameters!
# """)

# #---------------------------------#
# # Sidebar - Collects user input features into dataframe
# with st.sidebar.header('1. Upload your CSV data'):
#     uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
#     st.sidebar.markdown("""
# [Example CSV input file](https://raw.githubusercontent.com/dataprofessor/data/master/delaney_solubility_with_descriptors.csv)
# """)

# # Sidebar - Specify parameter settings
# with st.sidebar.header('2. Set Parameters'):
#     split_size = st.sidebar.slider('Data split ratio (% for Training Set)', 10, 90, 80, 5)

# with st.sidebar.subheader('2.1. Learning Parameters'):
#     parameter_n_estimators = st.sidebar.slider('Number of estimators (n_estimators)', 0, 1000, 100, 100)
#     parameter_max_features = st.sidebar.select_slider('Max features (max_features)', options=['auto', 'sqrt', 'log2'])
#     parameter_min_samples_split = st.sidebar.slider('Minimum number of samples required to split an internal node (min_samples_split)', 1, 10, 2, 1)
#     parameter_min_samples_leaf = st.sidebar.slider('Minimum number of samples required to be at a leaf node (min_samples_leaf)', 1, 10, 2, 1)

# with st.sidebar.subheader('2.2. General Parameters'):
#     parameter_random_state = st.sidebar.slider('Seed number (random_state)', 0, 1000, 42, 1)
#     parameter_criterion = st.sidebar.select_slider('Performance measure (criterion)', options=['mse', 'mae'])
#     parameter_bootstrap = st.sidebar.select_slider('Bootstrap samples when building trees (bootstrap)', options=[True, False])
#     parameter_oob_score = st.sidebar.select_slider('Whether to use out-of-bag samples to estimate the R^2 on unseen data (oob_score)', options=[False, True])
#     parameter_n_jobs = st.sidebar.select_slider('Number of jobs to run in parallel (n_jobs)', options=[1, -1])

# #---------------------------------#
# # Main panel

# # Displays the dataset
# st.subheader('1. Dataset')

# if uploaded_file is not None:
#     df = pd.read_csv(uploaded_file)
#     st.markdown('**1.1. Glimpse of dataset**')
#     st.write(df)
#     build_model(df)
# else:
#     st.info('Awaiting for CSV file to be uploaded.')
#     if st.button('Press to use Example Dataset'):
#         # Diabetes dataset
#         #diabetes = load_diabetes()
#         #X = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
#         #Y = pd.Series(diabetes.target, name='response')
#         #df = pd.concat( [X,Y], axis=1 )

#         #st.markdown('The Diabetes dataset is used as the example.')
#         #st.write(df.head(5))

#         # Boston housing dataset
#         boston = load_boston()
#         X = pd.DataFrame(boston.data, columns=boston.feature_names)
#         Y = pd.Series(boston.target, name='response')
#         df = pd.concat( [X,Y], axis=1 )

#         st.markdown('The Boston housing dataset is used as the example.')
#         st.write(df.head(5))

#         build_model(df)