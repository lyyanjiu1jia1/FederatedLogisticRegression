import pandas as pd


file_path = r'../data/'


def read_csv_to_dataframe(file_name):
    """

    :param file_name: str
    :return: dataframe
    """
    data = pd.read_csv(file_path + file_name, header=0)
    return data


def read_csv_to_ndarray(file_name):
    """

    :param file_name: str
    :return: ndarray
    """
    return read_csv_to_dataframe(file_name).to_numpy()


def assemble(guest_data_file_name, host_data_file_name, output_data_filename='total_data'):
    """

    :param guest_data_file_name: str
    :param host_data_file_name: str
    :param output_data_filename
    :return:
    """
    guest_df = read_csv_to_dataframe(guest_data_file_name)
    guest_new_column_names = []
    count = 0
    for column_name in guest_df.columns:
        if count < 2:
            guest_new_column_names.append(column_name)
            count += 1
        else:
            guest_new_column_names.append('guest_' + column_name)
    guest_df.columns = guest_new_column_names

    host_df = read_csv_to_dataframe(host_data_file_name)
    host_new_column_names = []
    count = 0
    for column_name in host_df.columns:
        if count < 1:
            host_new_column_names.append(column_name)
            count += 1
        else:
            host_new_column_names.append('host_' + column_name)
    host_df.columns = host_new_column_names

    # total_df = guest_df.set_index('date_id').join(host_df.set_index('date_key_md5'))
    total_df = pd.merge(guest_df, host_df, how='inner', on='date_id')
    total_df.to_csv(file_path + output_data_filename)


def disassemble(dataframe):
    """

    :param dataframe:
    :return:
    """
    y = dataframe['y'].to_numpy()
    y = y.reshape((y.shape[0], 1))

    x_dataframe = dataframe.drop(['y'], axis=1)
    X = x_dataframe.to_numpy()
    return X, y
