import logging as lg
import load_dataset


def preprocess():
    try:

        df = load_dataset()
        lg.info("Starting the data pre processing")

        # creating profile
        # lg.info("Starting the data profile")
        # profile = ProfileReport(df, explorative=True)
        # profile.to_file("AI4I_2020.html")
        # lg.info("data profile creation completed with name AI4I_2020.html and saved in ",os.getcwd())

        # Dropping Torque as Rotational is related (Multi collinear) to it within feature
        df.drop(['Torque [Nm]'], axis=1, inplace=True)

        # Dropping UDI as it is primary key
        df.drop(['UDI'], axis=1, inplace=True)

        # As Type column came from Product ID (first letter), we can drop Product ID
        df.drop(['Product ID'], axis=1, inplace=True)

        # Converting ordinal catergory (L, M, H) in Type to integers
        df['Type'] = df['Type'].str.replace('L', '1')
        df['Type'] = df['Type'].str.replace('M', '2')
        df['Type'] = df['Type'].str.replace('H', '3')

        # Changing column names for better appearance
        df.columns = ['Type', 'Air_temp', 'Process_temp', 'Rotaional_speed', 'Tool_wear', 'Machine_failure', 'TWF',
                      'HDF', 'PWF', 'OSF', 'RNF']

        # Replacing zeros values of df['Tool_wear'] to median as it is not normally distributed.
        df['Tool_wear'].median()
        df['Tool_wear'] = df['Tool_wear'].replace({0: '108'})

        # Converting object columns to int
        df['Tool_wear'] = df['Tool_wear'].astype(int)
        df['Type'] = df['Type'].astype(int)

        return df

    except Exception as e:
        lg.exception(str(e))
        lg.info("pre process step failed")
        print(str(e))

    else:
        lg.info("pre process step completed successfully")
