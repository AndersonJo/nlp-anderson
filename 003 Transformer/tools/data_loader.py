import pickle


def load_preprocessed_data(opt):
    batch_size = opt.batch_size
    data = pickle.load(open(opt.data_pkl, 'rb'))

    opt = data['opt']
    import ipdb
    ipdb.set_trace()