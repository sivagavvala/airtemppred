import logging as lg
import pickle


def save_model(linear):
    try:
        # Saving the model
        lg.info('saving the linear reg model')
        file = 'linear_reg_classwork.sav'
        pickle.dump(linear, open(file, 'wb'))

    except Exception as e:
        lg.info('model could not be saved')
        lg.exception(str(e))
        print(str(e))

    else:
        lg.info('model saved')
