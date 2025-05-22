import numpy as np
import pickle
import os
import os.path




def load_from_full_dataset(full_dataset_path, capture_date,rx_name,prefix=None):
    src=full_dataset_path

    if prefix is None: 
        dataset_path = '{}pkl_wifi_{}/dataset_{}_node{}.pkl'.format(src,capture_date,capture_date,rx_name)
    else:
        dataset_path = '{}pkl_wifi_{}_{}/dataset_{}_node{}.pkl'.format(src,prefix,capture_date,capture_date,rx_name)
   
    if os.path.isfile(dataset_path) :
        with open(dataset_path,'rb') as f:
            dataset = pickle.load(f)
    else:
            dataset = None
#             print('Not Found')
#             print(dataset_path)
    return dataset


def load_compact_pkl_dataset(dataset_path,dataset_name):
    with open(dataset_path+dataset_name+'.pkl','rb') as f:
        dataset = pickle.load(f)
    return dataset


def shuffle(vec1,vec2,seed = 0):
    np.random.seed(0)
#     print(vec1.shape[0],vec2.shape[0])
    shfl_indx = np.arange(vec1.shape[0])
    np.random.shuffle(shfl_indx)
    shfl_indx = shfl_indx.astype('int')
    vec1 = vec1[shfl_indx]
    vec2 = np.copy(vec2[shfl_indx])
    return vec1,vec2


def norm(sig_u):
    if len(sig_u.shape)==3:
        pwr = np.sqrt(np.mean(np.sum(sig_u**2,axis = -1),axis = -1))
        sig_u = sig_u/pwr[:,None,None]
    if len(sig_u.shape)==2:
        pwr =  np.sqrt(np.mean(np.sum(sig_u**2,axis = -1),axis = -1))
        sig_u = sig_u/pwr
    # print(sig_u.shape)
    return sig_u

def split3(vec,n1,n2):
    vec1 = vec[0:n1]
    vec2 = vec[n1:n1+n2]
    vec3 = vec[n1+n2:]
    return vec3,vec1,vec2

def split_set3(st,f1,f2):
    [sig,txid] = st

    n_samples  = sig.shape[0]
    n1 = int(f1*n_samples)
    n2 = int(f2*n_samples)

    sig1,sig2,sig3 = split3(sig,n1,n2)
    txid1,txid2,txid3 = split3(txid,n1,n2)
    st1 = [sig1,txid1]
    st2 = [sig2,txid2]
    st3 = [sig3,txid3]
    return st1,st2,st3 

def get_node_indices(tx_name_list,node_name_list):
    op_list = []
    for tx in tx_name_list:
        if tx in node_name_list:
            op_list.append(node_name_list.index(tx))
        else:
            op_list.append(None)
    return op_list
    
def parse_nodes(dataset,node_list,seed = 0):
    cat_sig = []
    cat_txid = []
    data = dataset['data']
    
    
    for i,node in enumerate(node_list):
        if (not node  is  None) and  node < len(data):
            cat_sig.append(data[node])
            cat_txid.append(np.ones( (data[node].shape[0]) )*i)
    cat_sig = np.concatenate(cat_sig)
    cat_txid = np.concatenate(cat_txid)
    np.random.seed(seed)
    cat_sig,cat_txid = shuffle(cat_sig,cat_txid)
    cat_sig = norm(cat_sig)
    return (cat_sig,cat_txid)

def to_categorical(y, num_classes=None, dtype='float32'):
    y = np.array(y, dtype='int')
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes), dtype=dtype)
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical

def prepare_txid_and_weights(st,n):
    sig,txid = st
    txid_oh = to_categorical(txid,n)
    stat= np.sum(txid_oh,axis=0)
    cls_weights = np.max(stat,axis=0)/stat 
    cls_weights = cls_weights.tolist()
    augset = [sig,txid,txid_oh,cls_weights]
    return augset

def prepare_dataset(dataset,tx_name_list,val_frac=0.1, test_frac=0.1):
    tx_list = get_node_indices(tx_name_list,dataset['node_list'])
    all_set = parse_nodes(dataset,tx_list,seed = 0)
    train_set,val_set,test_set = split_set3(all_set,val_frac, test_frac)
    train_augset = prepare_txid_and_weights(train_set,len(tx_list))
    val_augset = prepare_txid_and_weights(val_set,len(tx_list))
    test_augset = prepare_txid_and_weights(test_set,len(tx_list))
    return train_augset,val_augset,test_augset


def create_dataset_impl(tx_list,rx_list,capture_date_list,max_sig=None,equalized_list=[0],full_dataset_path = 'data/',op_dataset_file = None):
    dataset = {}
    dataset['tx_list'] = tx_list
    dataset['rx_list'] = rx_list
    dataset['capture_date_list']=capture_date_list
    dataset['equalized_list'] = equalized_list
    dataset['max_sig'] = max_sig
    
    n_tx = len(tx_list)
    n_rx = len(rx_list)
    n_day = len(capture_date_list)
    n_eq = len(equalized_list)
    
    prefix_lut = [None,'eq']
    
    prefix_list = [prefix_lut[tt] for tt in  equalized_list]
    
    dataset['data'] = [ [ [ [ [ ] for _ in range(n_eq)] for _ in range(n_day) ] for _ in range(n_rx) ]  for _ in range(n_tx)     ]
    
    
    missing_rx_dict = {}
    
    missing_files = False

    
    with open('IdSig_info.pkl','rb') as f:
        IdSig_info=pickle.load(f)
    
    slc = slice(None,max_sig)
    for day_i,capture_date in enumerate(capture_date_list):
        for rx_i,rx_train in enumerate(rx_list):
            for eq_i,prefix in enumerate(prefix_list):
                tdataset = load_from_full_dataset(full_dataset_path,capture_date,rx_train,prefix=prefix)
                if not tdataset is None:
                    for tx_i,tx in enumerate(tx_list):
                        if tx in tdataset['node_list']:
                            tx_indx = tdataset['node_list'].index(tx)
                            dataset['data'][tx_i][rx_i][day_i][eq_i]= tdataset['data'][tx_indx][slc]  
                        else:
                            dataset['data'][tx_i][rx_i][day_i][eq_i]=np.zeros((0,256,2))
                else:
                    missing_rx_name =rx_list[rx_i]  
                    eq_val = equalized_list[eq_i]
                    IdSig_info_sub  = IdSig_info[eq_val][capture_date]
                    if missing_rx_name  in IdSig_info_sub.keys():
                            missing_files = True
                            if not eq_val in  missing_rx_dict.keys():
                                missing_rx_dict[eq_val]={}
                            if not capture_date in  missing_rx_dict[eq_val].keys():
                                missing_rx_dict[eq_val][capture_date]=[]
                            missing_rx_info  = IdSig_info_sub[missing_rx_name]
                            missing_rx_dict[eq_val][capture_date].append(   (missing_rx_info['name'], missing_rx_info['link'],missing_rx_info['size']) )

    
    if missing_files:
        ii=1
        total_file_sizes = 0
        print('You have missing files that you need to download.')
        
        for eq_k  in missing_rx_dict.keys():  
            if len(missing_rx_dict[eq_val])>0:
                print('')
                if eq_k==0:
                    print('You need to download the following files for the non equalized dataset')
                else:
                    print('You need to download the following files for the equalized dataset')
                
                print('')
                
                for date_k  in missing_rx_dict[eq_k].keys():  
                    for missing_rx in missing_rx_dict[eq_val][date_k]:
                        print('{}) Name: {} , Size: {} MB'.format(ii,missing_rx[0],missing_rx[2]/1e6))
                        total_file_sizes=total_file_sizes+missing_rx[2]
                        ii=ii+1
                print('Links:')
                for date_k  in missing_rx_dict[eq_k].keys():  
                    for missing_rx in missing_rx_dict[eq_val][date_k]:
                        print('https://drive.google.com/u/0/uc?export=download&id={}'.format(missing_rx[1]))               
        print('')
        print('You need to dowlnoad {} GB'.format(total_file_sizes/1e9))
        print('Note the following:')
        print('1) The non-equalized and eqalized files need to be downloaded in different fodlers because they share the same exact names')
        print('2) The  non-equalized folders needs to be grouped by date and equalization using the same structure as the following google drive folder')
        print('https://drive.google.com/drive/folders/1r8cd4zZ7fwvN_iiyI_uDKbIFGZve49lw?usp=sharing')
        print('3) If you have already downloaded the files make sure that the full dataset path is configured correctly.')
        dataset = None
    else:
        if not op_dataset_file is None:
            with open(op_dataset_file,'wb') as f:
                pickle.dump(dataset,f)
                print('Dataset saved in {}'.format(op_dataset_file))

    return dataset


def merge_compact_dataset(compact_dataset,capture_date,tx_list,rx_list,max_sig=None,equalized=0):
    dataset = {}
    dataset['node_list'] = tx_list
    dataset['data'] = [ () for _ in range(len(tx_list))]
    
    if not type(capture_date) is list: 
        capture_date_list = [capture_date]
    else:
        capture_date_list = capture_date
    slc = slice(None,max_sig)
    for capture_date in capture_date_list:
        for rx_train in rx_list:
            for indx,tx in enumerate(tx_list):
                tx_i=compact_dataset['tx_list'].index(tx)
                rx_i=compact_dataset['rx_list'].index(rx_train)
                date_i=compact_dataset['capture_date_list'].index(capture_date)
                eq_i=compact_dataset['equalized_list'].index(equalized)
                dataset['data'][indx]  +=  (compact_dataset['data'][tx_i][rx_i][date_i][eq_i][slc],)
    for indx in range(len(tx_list)):
        if len(dataset['data'][indx])>0:
            dataset['data'][indx] =  np.concatenate(dataset['data'][indx])
        else:
            dataset['data'][indx] =np.zeros((0,256,2))
    return dataset

def generate_tx_datasets(compact_dataset, capture_date, tx_list, rx_list, max_sig=None, equalized=0):
    # 用于存储按发射机分组的数据集
    dataset = {}
    
    # 设置发射机列表（tx_list）作为节点列表
    dataset['node_list'] = tx_list
    
    # 初始化数据，每个发射机一个空元组
    dataset['data'] = [() for _ in range(len(tx_list))]

    # 如果 capture_date 不是列表，转化为列表
    if not isinstance(capture_date, list):
        capture_date_list = [capture_date]
    else:
        capture_date_list = capture_date
    
    # 创建切片对象（用于限制数据长度）
    slc = slice(None, max_sig)

    # 遍历 capture_date_list（可能包含多个日期）
    for capture_date in capture_date_list:
        # 遍历接收机列表（rx_list）
        for rx_train in rx_list:
            # 遍历发射机列表（tx_list）
            for indx, tx in enumerate(tx_list):
                # 查找发射机、接收机和日期在 compact_dataset 中的索引
                tx_i = compact_dataset['tx_list'].index(tx)
                rx_i = compact_dataset['rx_list'].index(rx_train)
                date_i = compact_dataset['capture_date_list'].index(capture_date)
                eq_i = compact_dataset['equalized_list'].index(equalized)

                # 提取对应的数据，并将其添加到 dataset['data'][indx] 中
                dataset['data'][indx] += (compact_dataset['data'][tx_i][rx_i][date_i][eq_i][slc],)
    
    # 对每个发射机的相关数据进行合并或填充空数据
    for indx in range(len(tx_list)):
        if len(dataset['data'][indx]) > 0:
            dataset['data'][indx] = np.concatenate(dataset['data'][indx])  # 合并数据
        else:
            # 如果没有数据，填充为零数组
            dataset['data'][indx] = np.zeros((0, 256, 2))

    return dataset

def preprocess_dataset_for_classification(compact_dataset, tx_list, rx_list, train_dates, max_sig=None, equalized=0):
    def extract_samples(dates):
        X = []
        y = []
        for rx in rx_list:
            for tx_idx, tx in enumerate(tx_list):
                tx_i = compact_dataset['tx_list'].index(tx)
                rx_i = compact_dataset['rx_list'].index(rx)
                eq_i = compact_dataset['equalized_list'].index(equalized)
                
                for date in dates:
                    if date not in compact_dataset['capture_date_list']:
                        continue
                    date_i = compact_dataset['capture_date_list'].index(date)
                    sig_data = compact_dataset['data'][tx_i][rx_i][date_i][eq_i]

                    if max_sig is not None:
                        sig_data = sig_data[:max_sig]

                    # 拆分成多个 (256, 2) 样本
                    for sample in sig_data:
                        if sample.shape == (256, 2):
                            X.append(sample)
                            y.append(tx_idx)
        
        return np.array(X), np.array(y)

    # 所有日期
    all_dates = set(compact_dataset['capture_date_list'])
    train_dates = set(train_dates)
    test_dates = list(all_dates - train_dates)

    # 转为 list 保持顺序
    X_train, y_train = extract_samples(list(train_dates))
    X_test, y_test = extract_samples(test_dates)

    print(f"✅ 训练样本数: {len(X_train)}, 测试样本数: {len(X_test)}")
    return X_train, y_train, X_test, y_test

def preprocess_dataset_cross_IQ_blocks(compact_dataset, tx_list, rx_list, train_dates, max_sig=None, equalized=0, block_size=250):
    def extract_samples(dates):
        X = []
        y = []
    
        for rx in rx_list:
            for tx_idx, tx in enumerate(tx_list):
                tx_i = compact_dataset['tx_list'].index(tx)
                rx_i = compact_dataset['rx_list'].index(rx)
                eq_i = compact_dataset['equalized_list'].index(equalized)

                for date in dates:
                    if date not in compact_dataset['capture_date_list']:
                        continue
                    date_i = compact_dataset['capture_date_list'].index(date)

                    sig_data = compact_dataset['data'][tx_i][rx_i][date_i][eq_i]  # shape: (N, 256, 2)

                    if max_sig is not None:
                        sig_data = sig_data[:max_sig]

                    num_signals = len(sig_data)
                    num_blocks = num_signals // block_size

                    for i in range(num_blocks):
                        block = sig_data[i * block_size : (i + 1) * block_size]  # (250, 256, 2)
                        if block.shape != (block_size, 256, 2):
                            continue

                        # 转置为 (256, 250, 2)
                        block_transposed = block.transpose(1, 0, 2)

                        # ✅ 拆成 256 个样本，每个是 (250, 2)
                        for j in range(block_transposed.shape[0]):
                            sample = block_transposed[j]  # shape: (250, 2)
                            X.append(sample)
                            y.append(tx_idx)

        return np.array(X), np.array(y)

    X_train, y_train = extract_samples(train_dates)
    test_dates = [d for d in compact_dataset['capture_date_list'] if d not in train_dates]
    X_test, y_test = extract_samples(test_dates)

    return X_train, y_train, X_test, y_test


def preprocess_dataset_cross_IQ_blocks_grouped_rx_fine_grained(compact_dataset, tx_list, rx_list, train_dates, max_sig=None, equalized=0, block_size=250, y=10):
    def extract_samples(dates):
        X = []
        y_labels = []

        for tx_idx, tx in enumerate(tx_list):
            tx_i = compact_dataset['tx_list'].index(tx)
            eq_i = compact_dataset['equalized_list'].index(equalized)

            for date in dates:
                if date not in compact_dataset['capture_date_list']:
                    continue
                date_i = compact_dataset['capture_date_list'].index(date)

                # === 获取每个 rx 的信号 ===
                rx_signals = []
                min_len = float('inf')

                for rx in rx_list:
                    rx_i = compact_dataset['rx_list'].index(rx)
                    sig_data = compact_dataset['data'][tx_i][rx_i][date_i][eq_i]  # shape: (N, 256, 2)

                    if max_sig is not None:
                        sig_data = sig_data[:max_sig]

                    rx_signals.append(sig_data)
                    min_len = min(min_len, len(sig_data))

                # === 计算能分多少个 step，每个 step 每个 rx 取 y 条 ===
                num_chunks = min_len // y
                combined_data = []

                for chunk_idx in range(num_chunks):
                    for rx_data in rx_signals:
                        start = chunk_idx * y
                        end = start + y
                        combined_data.append(rx_data[start:end])  # shape: (y, 256, 2)

                # 拼接成一个大的数组: shape (num_chunks * len(rx_list) * y, 256, 2)
                combined_data = np.concatenate(combined_data, axis=0)

                # === 分 block、转置、拆样本 ===
                num_signals = len(combined_data)
                num_blocks = num_signals // block_size

                for i in range(num_blocks):
                    block = combined_data[i * block_size : (i + 1) * block_size]  # shape: (block_size, 256, 2)

                    if block.shape != (block_size, 256, 2):
                        continue

                    block_transposed = block.transpose(1, 0, 2)  # shape: (256, block_size, 2)

                    for j in range(block_transposed.shape[0]):
                        sample = block_transposed[j]  # shape: (block_size, 2)
                        X.append(sample)
                        y_labels.append(tx_idx)

        return np.array(X), np.array(y_labels)

    X_train, y_train = extract_samples(train_dates)
    test_dates = [d for d in compact_dataset['capture_date_list'] if d not in train_dates]
    X_test, y_test = extract_samples(test_dates)

    return X_train, y_train, X_test, y_test
