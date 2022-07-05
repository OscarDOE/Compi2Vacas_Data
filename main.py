from sklearn import preprocessing
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.naive_bayes import GaussianNB;
from sklearn.metrics import mean_squared_error, r2_score

st.set_page_config(page_title='Data Science - Compiladores 2',
    layout='wide')

st.write("""# Compiladores 2 - Data Science """)
archi1, archi2, archi3 = st.columns([2,2,2])
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
                    func_tend += "+"+str(c)+"x^"+str(contequis)
                    contequis += 1
                else:
                    func_tend += str(c)+"x^"+str(contequis)
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
    print("GAUSSIANA")
    data = preprocessing.LabelEncoder()
    features = []
    for d in archivo:
        lista = tuple(data.fit_transform(archivo[d]))
        features.append(lista)
    
    features_encoded_antes = list()
    for i in range(len(features[0])):
        arr = [fila[i] for fila in features]
        features_encoded_antes.append(tuple(arr))
    with tablagauss2:
        st.write("Datos Codificados")
        st.dataframe(features_encoded_antes)
    booleanos = features.pop()
    features_encoded = list()
    for i in range(len(features[0])):
        arr = [fila[i] for fila in features]
        features_encoded.append(tuple(arr))
    

    Y_predict = st.text_input("Ingrese el término a predecir, en términos de los datos codificados")
    if Y_predict == "":
        st.info("Ingrese el valor a predecir")
    
    
    model = GaussianNB()
    model.fit(features_encoded, booleanos)
    print(model)

    if Y_predict != "":
        Y_pred = Y_predict.strip()
        array_predict = list()
        for i in Y_pred:
            if i != "," and i != " " and i != ", ":
                array_predict.append(int(i))
        predict = model.predict([array_predict])
        predicto = arr
        st.header("El valor es: "+ str(predict[0]))


def arbolito(archivo):
    data = preprocessing.LabelEncoder()
    Y_predict = st.text_input(
            'Introduce los valores a clasificar')
    features = []
    for d in archivo:
        lista = tuple(data.fit_transform(archivo[d]))
        features.append(lista)
    features_encoded_antes = list()
    for i in range(len(features[0])):
        arr = [fila[i] for fila in features]
        features_encoded_antes.append(tuple(arr))
    with tablagauss2:
        st.write("Datos Codificados")
        st.dataframe(features_encoded_antes)
    if Y_predict != "":
        booleanos = features.pop()
        features_encoded = list()
        for i in range(len(features[0])):
            arr = [fila[i] for fila in features]
            features_encoded.append(tuple(arr))
        Y_pred = Y_predict.strip()
        array_predict = list()
        for i in Y_pred:
            if i != "," and i != " " and i != ", ":
                array_predict.append(int(i))
        model = DecisionTreeClassifier().fit(features_encoded, booleanos)
        
        plot_tree(model, filled=True)
        plt.savefig("Arbol.png")
        st.subheader("Grafica: ")
        st.image("Arbol.png")
        if Y_predict != "":
            predict = model.predict([array_predict])
            st.info(predict[0])   
    else:
        st.info("INTRODUZCA VALORES DE PRUEBA")

def redes(archivo):
    st.write("REDES")
    data = preprocessing.LabelEncoder()
    Y_predict = st.text_input(
            'Introduce los valores a clasificar')
    features = []
    for d in archivo:
        lista = tuple(data.fit_transform(archivo[d]))
        features.append(lista)
    features_encoded_antes = list()
    for i in range(len(features[0])):
        arr = [fila[i] for fila in features]
        features_encoded_antes.append(tuple(arr))
    with tablagauss2:
        st.write("Datos Codificados")
        st.dataframe(features_encoded_antes)

    if Y_predict != "":
        booleanos = features.pop()
        features_encoded = list()
        for i in range(len(features[0])):
            arr = [fila[i] for fila in features]
            features_encoded.append(tuple(arr))
        Y_pred = Y_predict.strip()
        array_predict = list()
        for i in Y_pred:
            if i != "," and i != " " and i != ", ":
                array_predict.append(int(i))
        model = MLPClassifier(hidden_layer_sizes=(10,10,10), max_iter=500, alpha=0.0001,
                     solver='adam', random_state=21,tol=0.000000001).fit(features_encoded, booleanos)

        if Y_predict != "":
            predict = model.predict([array_predict])
            st.info(predict[0])   
    else:
        st.info("INTRODUZCA VALORES DE PRUEBA")

    

with archi1:
    st.header("--- Introduzca su archivo")
    carga = st.file_uploader("Archivo de entrada", type=["csv","json","xls","xlsx"])
with archi2:
    st.header("Seleccione la función a realizar")
    opcion = st.selectbox(
      'Choose an algorithm:',
     ('','Regresión lineal', 'Regresión polinomial', 'Clasificador Gaussiano','Clasificador de árboles de decisión','Redes neuronales'))
if carga is not None:
    st.markdown('Tabla de Datos')
    print("TIPO")
    print(carga.type)
    if carga.type == "text/csv":
        data = pd.read_csv(carga)
    elif carga.type == "application/vnd.ms-excel":
        data = pd.read_excel(carga)
    elif carga.type == "application/json":
        data = pd.read_json(carga)
    elif carga.type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
        data = pd.read_excel(carga)
    else:
        st.warning("No es un tipo de archivo válido, solo se permiten 'csv', 'xls', 'xlsx' y 'json'")
        
    tablagauss1, tablagauss2 = st.columns([2,2])
    with tablagauss1:
        st.write(data)
    if(opcion != ""):
        if opcion == "Regresión lineal":
            lineal(data)
        if opcion == "Regresión polinomial":
            polinomial(data)
        if opcion == "Clasificador Gaussiano":
            gauss(data)
        if opcion == "Clasificador de árboles de decisión":
            arbolito(data)
        if opcion == "Redes neuronales":
            redes(data)
        
else:
    st.info('Introduzca el archivo para continuar...')