import logging as lg

from flask import Flask, render_template, request
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso, RidgeCV, LassoCV, ElasticNet, ElasticNetCV, LinearRegression
from sklearn.model_selection import train_test_split
from load_dataset import preprocess_dataset
import numpy as np
from model_predict import model_predict

app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])  # To render Homepage
def home_page():
    return render_template('index.html')


@app.route('/dbaction', methods=['GET', 'POST'])  # This will be called from UI
def math_operation():
    if request.method == 'POST':
        operation = request.form['operation']

        try:
            lg.basicConfig(filename='linear1.log', level=lg.DEBUG,
                           format=' %(asctime)s -%(levelname)s -  %(message)s')

            df = preprocess_dataset()

            # Creating dependent and independent variables

            x = df[
                ['Type', 'Process_temp', 'Rotaional_speed', 'Tool_wear', 'Machine_failure', 'TWF', 'HDF', 'PWF', 'OSF',
                 'RNF']]
            # x = df.drop("Air_temp")
            y = df['Air_temp']

            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=100)

            sc = StandardScaler()
            x_train = sc.fit_transform(x_train)
            x_test = sc.transform(x_test)

            ################################################################

            if operation == 'simple_lr':

                # Creating linear regression object
                linear = LinearRegression()
                lg.info("Model Object created successfully")

                # Fitting linear reg obj with our data set
                linear.fit(x_train, y_train)
                lg.info("Model fit done successfully")

                # Predicting Air_temp with new entry from our fitted model
                lg.info("Predicting sample entry")
                result1 = model_predict(linear)
                lg.info("sample entry predicted with simple LR ")

                def adj_r2(x, y):
                    r2 = linear.score(x, y)
                    n = x.shape[0]
                    p = x.shape[1]
                    adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
                    return adjusted_r2

                # Accuracy of LR model
                acc = linear.score(x_test, y_test)

                # return render_template('results.html',  tables=[df.to_html(classes='data')], titles=df.columns.values)
                return render_template('results.html', result=acc)

            else:
                lg.info("linear regression model created")

            ##########################################################

            if operation == 'lasso_lr':

                # Creating linear regression object
                lassocv = LassoCV(alphas=None, cv=50, max_iter=200000, normalize=True)
                lassocv.fit(x_train, y_train)
                lg.info("Model Object created successfully")

                # Fitting Lasso reg obj with our data set
                lasso = Lasso(alpha=lassocv.alpha_)
                lasso.fit(x_train, y_train)
                lg.info("Model fit done successfully")

                # Predicting Air_temp with new entry from our fitted model
                lg.info("Predicting sample entry")
                result1 = model_predict(lasso)
                lg.info("sample entry predicted with Lasso LR ")

                def adj_r2(x, y):
                    r2 = lasso.score(x, y)
                    n = x.shape[0]
                    p = x.shape[1]
                    adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
                    return adjusted_r2

                # Accuracy of LR model
                acc = lasso.score(x_test, y_test)

                # return render_template('results.html',  tables=[df.to_html(classes='data')], titles=df.columns.values)
                return render_template('results.html', result=acc)

            else:
                lg.info("Lasso linear regression model created")

            ################################################################

            if operation == 'ridge_lr':

                # Creating Ridge regression object
                ridgecv = RidgeCV(alphas=np.random.uniform(0, 10, 50), cv=50, normalize=True)
                ridgecv.fit(x_train, y_train)
                lg.info("Model Object created successfully")

                # Fitting Ridge reg obj with our data set
                ridge = Ridge(alpha=ridgecv.alpha_)
                ridge.fit(x_train, y_train)
                lg.info("Model fit done successfully")

                # Predicting Air_temp with new entry from our fitted model
                lg.info("Predicting sample entry")
                result1 = model_predict(ridge)
                lg.info("sample entry predicted with Ridge LR ")

                def adj_r2(x, y):
                    r2 = ridge.score(x, y)
                    n = x.shape[0]
                    p = x.shape[1]
                    adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
                    return adjusted_r2

                # Accuracy of LR model
                acc = ridge.score(x_test, y_test)

                # return render_template('results.html',  tables=[df.to_html(classes='data')], titles=df.columns.values)
                return render_template('results.html', result=acc)

            else:
                lg.info("Ridge linear regression model created")

            ################################################################

            if operation == 'elastic_lr':

                # Creating Ridge regression object
                elasticcv = ElasticNetCV(alphas=None, cv=10)
                elasticcv.fit(x_train, y_train)
                lg.info("Model Object created successfully")

                # Fitting Lasso reg obj with our data set
                elastic = ElasticNet(alpha=elasticcv.alpha_, l1_ratio=elasticcv.l1_ratio_)
                elastic.fit(x_train, y_train)
                lg.info("Model fit done successfully")

                # Predicting Air_temp with new entry from our fitted model
                lg.info("Predicting sample entry")
                result1 = model_predict(elastic)
                lg.info("sample entry predicted with simple LR ")

                def adj_r2(x, y):
                    r2 = elastic.score(x, y)
                    n = x.shape[0]
                    p = x.shape[1]
                    adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
                    return adjusted_r2

                # Accuracy of LR model
                acc = elastic.score(x_test, y_test)

                # return render_template('results.html',  tables=[df.to_html(classes='data')], titles=df.columns.values)
                return render_template('results.html', result=acc)

            else:
                lg.info("Elastic linear regression model created")

        except Exception as e:
            lg.info('could not create linear object model')
            lg.exception(str(e))
            print(str(e))


if __name__ == '__main__':
    app.run()
