from shutil import which
from sklearn import preprocessing
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.naive_bayes import GaussianNB;
from sklearn.metrics import mean_squared_error, r2_score


#---------------------------------#
#Global variables


#---------------------------------#



#---------------------------------#
# Page layout
## Page expands to full width
st.set_page_config(page_title='Data Science - Compiladores 2',
    layout='wide')

#---------------------------------#



#---------------------------------#
st.write("""
# Compiladores 2 - Data Science
""")
archi1, archi2, archi3 = st.columns([2,2,2])

#---------------------------------#
# Displays the dataset
def lineal(archivo):
    X = []
    var = []
    var.append(' ')
    for d in archivo.columns:
        var.append(d)
    with st.sidebar.header('EJE X'):
            valorX = st.sidebar.selectbox(
            'Variable X',    
            (var))
    with st.sidebar.header('EJE Y'):
        valorY = st.sidebar.selectbox(
        'Variable Y',
        (var))
    if valorY != " " and valorX != " ":
        X = np.asarray(archivo[valorX]).reshape(-1, 1)
        Y = data[valorY]
        linear_regression = LinearRegression()
        linear_regression.fit(X, Y)
        Y_pred = linear_regression.predict(X)

        col1, col2, col3, col4 = st.columns([2,2,5,3])
        with col1:
            st.write('Variable X :')
            st.write(valorX)
            st.write(X)
        with col2:
            st.write('Variable Y : ')
            st.write(valorY)
            st.write(Y_pred)

        with col4:
            st.write("")
            st.write("")
            st.write("")
            st.write("")
            st.write("Error medio: ")
            st.info(mean_squared_error(Y, Y_pred, squared=True))
            st.write("Coeficiente: ")
            st.info(linear_regression.coef_)
            st.write("R2: ")
            st.info(r2_score(Y, Y_pred))


        with st.sidebar.header(''):
            st.header("Ingrese el valor a predecir: ")
            title = st.text_input('Ingrese el valor a predecir: ')

        plt.scatter(X, Y)
        plt.plot(X, Y_pred, color='red')
        plt.savefig("graficaLineal.png")
        with col3:
            st.image("./graficaLineal.png")
        coe = linear_regression.coef_[0]
        inter=linear_regression.intercept_
        coe2="{0:.4f}".format(coe)
        inter2="{0:.4f}".format(inter)
        col1, col2, col3 = st.columns([4,3,3])
        with col1:
            st.write("")
        with col2:
            st.header("Funcion de tendencia ")
        with col3:
            st.write("")
        if float(coe2)>0:
            st.latex(f'''y = {str(inter2)} + {str(coe2)} x''')
        else: st.latex(f'''y = {str(inter2)} {str(coe2)} x''')        
        if title != "":
            titleInt = int(title)
            st.write("Predicción: ")
            Y_new = linear_regression.predict([[int(titleInt)]])
            st.info(Y_new)
        else: st.info('Ingrese un valor a predecir...')
    else:st.info('Seleccione las variables X y Y...')

def polinomial(archivo):
    with archi3:
        st.header("Grado")
        grad = st.number_input("Seleccione el grado")
        grade = int(grad)
    X = []
    var = []
    var.append(' ')
    for d in archivo.columns:
        var.append(d)
    with st.sidebar.header('EJE X'):
            valorX = st.sidebar.selectbox(
            'Variable X',    
            (var))
    with st.sidebar.header('EJE Y'):
        valorY = st.sidebar.selectbox(
        'Variable Y',
        (var))
    if valorY != " " and valorX != " " and grade != " ":
        if grade != " ":
            X = np.asarray(archivo[valorX]).reshape(-1, 1)
            Y = data[valorY]
            poly_reg = PolynomialFeatures(degree = grade)
            X_poly = poly_reg.fit_transform(X)
            linear_regression = LinearRegression()
            linear_regression.fit(X_poly, Y)
            Y_pred = linear_regression.predict(X_poly)
            col1, col2, col3, col4 = st.columns([2,2,5,3])
            with col1:
                st.write('Variable X :')
                st.write(valorX)
                st.write(X)
            with col2:
                st.write('Variable Y : ')
                st.write(valorY)
                st.write(Y_pred)

            with col4:
                st.write("")
                st.write("")
                st.write("")
                st.write("")
                st.write("Error medio: ")
                st.info(mean_squared_error(Y, Y_pred, squared=False))
                st.write("Coeficiente: ")
                st.info(linear_regression.coef_[1])
                st.write("R2: ")
                #st.latex('''R ^2''')
                st.info(r2_score(Y, Y_pred))

            with st.sidebar.header(''):
                st.header("Ingrese el valor a predecir: ")
                title = st.text_input('Ingrese el valor a predecir: ')

            plt.scatter(X, Y, color='red')
            if(grade > 1):
                X_grid = np.arange(min(X), max(X),0.1)
                X_grid = X_grid.reshape(len(X_grid),1)
                plt.plot(X_grid, linear_regression.predict(poly_reg.fit_transform(X_grid)), color='blue')
            elif(grade == 1):
                linear_regression = LinearRegression()
                linear_regression.fit(X, Y)
                plt.plot(X, Y_pred, color='red')
            else:
                pass
            plt.title("Regresion Polinomial de grado "+str(grade))
            plt.xlabel(valorX)
            plt.ylabel(valorY)
            plt.savefig("graficaLineal.png")
            with col3:
                st.image("./graficaLineal.png")
            coe = linear_regression.coef_[0]
            inter=linear_regression.intercept_
            inter2="{0:.4f}".format(inter)
            col1, col2, col3 = st.columns([4,3,3])
            with col1:
                st.write("")
            with col2:
                st.header("Funcion de tendencia ")
            with col3:
                st.write("")
            func_tend = ""
            contequis = 1
            for d in linear_regression.coef_:
                c = "{0:.4f}".format(d)
                if float(d) == 0:
                    pass
                elif float(d) > 0:
                    func_tend += "+"+str(c)+"x"+str(contequis)
                    contequis += 1
                else:
                    func_tend += str(c)+"x"+str(contequis)
                    contequis += 1
            st.latex(f'''y = {str(inter2)}  {func_tend} ''')
            if title != "":
                titleInt = int(title)
                st.write("Predicción: ")

                Y_new = linear_regression.predict(poly_reg.fit_transform([[int(titleInt)]]))
                st.info(Y_new)
                
            else:
                st.info('Ingrese un valor a predecir...')
        else:
            st.info('Ingrese un grado para las funciones')
    else:
        st.info('Seleccione las variables X y Y...')


def gauss(archivo):
    Y_predict = st.text_input("Ingrese el dato a predecir")
    if Y_predict == "":
        st.info("Ingrese el valor a predecir")
    data = preprocessing.LabelEncoder()
    features = []
    for d in archivo:
        lista = tuple(data.fit_transform(archivo[d]))
        features.append(lista)
    
    booleanos = features.pop()
    features_encoded = list()

    for i in range(len(features[0])):
        arr = [fila[i] for fila in features]
        features_encoded.append(tuple(arr))
    print(features_encoded)
    model = GaussianNB()
    model.fit(features_encoded, booleanos)

    if Y_predict != "":
        Y_pred = Y_predict.strip()
        array_predict = list()
        for i in Y_pred:
            if i != "," and i != " " and i != ", ":
                array_predict.append(int(i))
        predict = model.predict([array_predict])
        st.write(predict[0])

#---------------------------------#
# Sidebar - Collects user input features into dataframeº
#with st.sidebar.header('Seleccione su archivo con extensión .CSV'):

with archi1:
    st.header("--- Introduzca su archivo")
    carga = st.file_uploader("Archivo CSV", type=["csv"])
with archi2:
    st.header("Seleccione la función a realizar")
    opcion = st.selectbox(
      'Choose an algorithm:',
     ('','Regresión lineal', 'Regresión polinomial', 'Clasificador Gaussiano','Clasificador de árboles de decisión','Redes neuronales'))

#---------------------------------#

#---------------------------------#

if carga is not None:
    st.markdown('Tabla de Datos')
    data = pd.read_csv(carga)
    #st.markdown('**1.1. Data set overview**')
    st.write(data)

    if(opcion != ""):
        if opcion == "Regresión lineal":
            lineal(data)
        if opcion == "Regresión polinomial":
            polinomial(data)
        if opcion == "Clasificador Gaussiano":
            gauss(data)
else:
    st.info('Introduzca el archivo para continuar...')