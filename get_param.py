import logging as lg


def get_param(linear):
    lg.info('printing params of linear reg model')
    print("Intercept of our model is ", linear.intercept_)
    print("Coefficients of model are", linear.coef_)
