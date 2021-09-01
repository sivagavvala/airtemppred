import logging as lg


def model_predict(linear):
    lg.info("Predicting Air temp from user entry")
    # ip=input('enter Type(L=1,M=2,H=3),Process temperature [K],Rotational speed [rpm],Tool wear [min],
    # Machine failure,TWF,HDF,PWF,OSF,RNF') lg.info("user entered input is ",ip)
    try:

        lg.info("predicting the air temp...")
        # print("predicted air temp is...", linear.predict([[1, 308.5, 1600, 5, 0, 0, 0, 0, 0, 0]]))
        return linear.predict([[1, 308.5, 1600, 5, 0, 0, 0, 0, 0, 0]])
        lg.info("sample entry predicted")
    except Exception as e:
        lg.exception(str(e))
        lg.info("predicting step failed")
        print(str(e))
