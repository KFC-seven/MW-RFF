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

# 将X天的IQ信号作为训练集，其余天数作为测试集。然后现在不需要画图了，只需要原始的256个信号作为一条数据就行，然后加上这个发射机的标签。compact_dataset[‘data‘]的数据结构是（发射机数量, 接收机数量, 采集日期, 是否信道均衡化，（N,256,2） ）
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

# X_train和X_test的结构是（样本数，256,2）我现在想的是，按照每250条数据做为一个块（250，256,2），每个块里面做转置就变成了（256,250,2），这样就变成了IQ信号的跨序连接，因为对应的把每个采样点上IQ进行了连接，而不是原来的256个采样点顺序连接。
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

# 每个rx每天有Y条信号，但是我拼接的顺序是每个rx先拿出前y个信号进行拼接。然后再按顺序拿出y个信号。直到Y被拿完。
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

# 每个tx的每个日期进行层次交错，每个日期都有Y个信号，按日期顺序依次拿出y个信号放一起，和rx无关
def preprocess_dataset_cross_IQ_blocks_date_interleaved(compact_dataset, tx_list, train_dates, max_sig=None, equalized=0, block_size=250, y=10):
    def extract_samples(dates):
        X = []
        y_labels = []

        for tx_idx, tx in enumerate(tx_list):
            tx_i = compact_dataset['tx_list'].index(tx)
            eq_i = compact_dataset['equalized_list'].index(equalized)

            # === 为每个日期提取信号 ===
            date_signal_lists = []

            for date in dates:
                if date not in compact_dataset['capture_date_list']:
                    continue

                date_i = compact_dataset['capture_date_list'].index(date)
                all_rx_data = []

                # 将所有 rx 的信号拼接（无视 rx）
                for rx_i in range(len(compact_dataset['rx_list'])):
                    sig_data = compact_dataset['data'][tx_i][rx_i][date_i][eq_i]  # shape: (N, 256, 2)
                    if max_sig is not None:
                        sig_data = sig_data[:max_sig]
                    all_rx_data.append(sig_data)

                date_data = np.concatenate(all_rx_data, axis=0)  # shape: (total, 256, 2)
                np.random.shuffle(date_data)  # 打乱以避免一个 rx 局部占据前几条
                date_signal_lists.append(date_data)

            # === 交错采样 ===
            # 先找每个日期最少能提供几个 y-size 片段
            min_chunks = min(len(date_data) // y for date_data in date_signal_lists)

            combined_data = []
            for chunk_idx in range(min_chunks):
                for date_data in date_signal_lists:
                    start = chunk_idx * y
                    end = start + y
                    combined_data.append(date_data[start:end])  # shape: (y, 256, 2)

            combined_data = np.concatenate(combined_data, axis=0)  # shape: (min_chunks * len(dates) * y, 256, 2)

            # === 分 block、转置、拆样本 ===
            num_signals = len(combined_data)
            num_blocks = num_signals // block_size

            for i in range(num_blocks):
                block = combined_data[i * block_size : (i + 1) * block_size]  # (block_size, 256, 2)

                if block.shape != (block_size, 256, 2):
                    continue

                block_transposed = block.transpose(1, 0, 2)  # (256, block_size, 2)

                for j in range(block_transposed.shape[0]):
                    sample = block_transposed[j]  # shape: (block_size, 2)
                    X.append(sample)
                    y_labels.append(tx_idx)

        return np.array(X), np.array(y_labels)

    X_train, y_train = extract_samples(train_dates)
    test_dates = [d for d in compact_dataset['capture_date_list'] if d not in train_dates]
    X_test, y_test = extract_samples(test_dates)

    return X_train, y_train, X_test, y_test

def load_ltev_dataset(data_root, tx_list, rx_list, speeds):
    X = []
    y = []
    
    for tx_idx, tx in enumerate(tx_list):
        for rx in rx_list:
            for spd in speeds:
                # 构造文件路径，例如 data/TX1_RX1_10kmh.mat
                file_name = f"{tx}_{rx}_{spd}kmh.mat"
                file_path = os.path.join(data_root, file_name)
                
                if not os.path.exists(file_path):
                    print(f"文件 {file_path} 不存在，跳过")
                    continue
                
                mat_data = sio.loadmat(file_path)
                # 假设变量名是 'samples'，shape (288, N)
                samples = mat_data['samples']  # (288, 2999)
                
                # 转置成 (N, 288)
                samples = samples.T
                
                # 转成复数 (I, Q)
                I = samples.real
                Q = samples.imag
                iq_data = np.stack([I, Q], axis=-1)  # shape (N, 288, 2)
                
                X.append(iq_data)
                y.append(np.full((iq_data.shape[0],), tx_idx))
    
    X = np.concatenate(X, axis=0)
    y = np.concatenate(y, axis=0)
    
    return X, y


def preprocess_per_label(X_all, y_all, group_size=288):
    """
    对每个发射机类别分别进行随机打乱、分组、转置操作。
    
    X_all: np.array, shape (N, 288, 2), 所有信号数据
    y_all: np.array, shape (N,), 标签数组
    
    返回：
    X_new: np.array, shape (M, group_size, 2)
    y_new: np.array, shape (M,)
    """

    X_new_list = []
    y_new_list = []

    unique_labels = np.unique(y_all)

    for label in unique_labels:
        # 选出该标签对应的数据
        idxs = np.where(y_all == label)[0]
        X_label = X_all[idxs]
        y_label = y_all[idxs]

        # 打乱该类别数据
        perm = np.random.permutation(len(X_label))
        X_label = X_label[perm]
        y_label = y_label[perm]

        total_signals = len(X_label)
        num_groups = total_signals // group_size
        truncated_len = num_groups * group_size

        if truncated_len == 0:
            # 该类别数据太少，不足一组，跳过
            continue

        X_label_trunc = X_label[:truncated_len]
        y_label_trunc = y_label[:truncated_len]

        # 重塑为组结构
        X_groups = X_label_trunc.reshape(num_groups, group_size, 288, 2)

        # 转置采样点和信号条数维度
        X_groups_t = np.transpose(X_groups, (0, 2, 1, 3))

        # 合并组和采样点维度，形成新样本
        X_new_label = X_groups_t.reshape(num_groups * 288, group_size, 2)

        # 标签扩展：每组标签取组内第一个信号标签，重复288次
        y_groups = y_label_trunc.reshape(num_groups, group_size)
        y_new_label = []
        for group_labels in y_groups:
            y_new_label.extend([group_labels[0]] * 288)
        y_new_label = np.array(y_new_label)

        # 收集
        X_new_list.append(X_new_label)
        y_new_list.append(y_new_label)

    # 拼接所有类别的数据
    X_new = np.concatenate(X_new_list, axis=0)
    y_new = np.concatenate(y_new_list, axis=0)

    print(f"总样本数：{len(X_all)}, 处理后新样本数：{X_new.shape[0]}, 每样本序列长度：{X_new.shape[1]}")

    return X_new, y_new

def preprocess_per_label_by_file_order(X_list, y_list, group_size=288):
    """
    按发射机类别，按文件顺序依次取每个文件前 group_size/文件数 的数据拼接，
    对拼接后的数据做采样点转置及展开处理，形成新的训练样本。

    X_list: list of np.array, 每个元素是一个文件的信号，形状 (num_samples_per_file, 288, 2)
    y_list: list of labels，长度和X_list一致，表示每个文件的发射机标签
    group_size: int, 每个最终样本包含多少信号条数（对应你之前288）

    返回：
    X_new: np.array, (新的样本数量, group_size, 2)
    y_new: np.array, (新的样本数量,)
    """
    from collections import defaultdict

    # 按类别收集对应的文件索引和数据
    label_files = defaultdict(list)
    for i, label in enumerate(y_list):
        label_files[label].append(X_list[i])

    X_new_list = []
    y_new_list = []

    for label, files in label_files.items():
        num_files = len(files)
        # 每个文件取样数
        samples_per_file = group_size // num_files

        # 每个文件截取samples_per_file个信号
        truncated_files = []
        for file_data in files:
            if file_data.shape[0] < samples_per_file:
                # 如果文件样本不够，直接舍弃该类别（或者也可以舍弃这个文件，这里简单跳过）
                break
            truncated_files.append(file_data[:samples_per_file])

        if len(truncated_files) != num_files:
            # 某文件样本不足，跳过该类别
            continue

        # 拼接所有文件截断后的数据，按顺序排列，shape = (group_size, 288, 2)
        X_concat = np.concatenate(truncated_files, axis=0)

        # 现在开始做采样点转置操作：
        # 先按 group_size 分组，组数是 1 （因为总长度就是group_size）
        # reshape成 (1, group_size, 288, 2)
        X_group = X_concat.reshape(1, group_size, 288, 2)
        # 转置采样点和信号条数维度，变成 (1, 288, group_size, 2)
        X_group_t = np.transpose(X_group, (0, 2, 1, 3))
        # 合并组和采样点维度，变成 (1 * 288, group_size, 2)
        X_final = X_group_t.reshape(group_size * 288, group_size, 2)

        # 标签扩展，group_size*288个样本，标签是该类别重复
        y_final = np.array([label] * (group_size * 288))

        X_new_list.append(X_final)
        y_new_list.append(y_final)

    # 拼接所有类别数据
    if len(X_new_list) == 0:
        return np.array([]), np.array([])

    X_new = np.concatenate(X_new_list, axis=0)
    y_new = np.concatenate(y_new_list, axis=0)

    print(f"总文件数: {len(X_list)}, 处理后样本数: {X_new.shape[0]}, 每样本序列长度: {X_new.shape[1]}")

    return X_new, y_new

def preprocess_dataset_for_classification_with_diff(compact_dataset, tx_list, rx_list, train_dates, max_sig=None, equalized=0, use_differential=False):
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
                            # 应用差分功能
                            if use_differential:
                                sample = apply_differential(sample)
                            
                            X.append(sample)
                            y.append(tx_idx)
        
        return np.array(X), np.array(y)
    
    def apply_differential(signal):
        """
        对IQ信号应用差分操作
        signal: 形状为 (256, 2) 的数组，其中第0列是I分量，第1列是Q分量
        返回: 差分后的信号，形状与输入相同
        """
        # 计算差分：x_diff[n] = x[n] * conj(x[n-1])
        iq_complex = signal[:, 0] + 1j * signal[:, 1]  # 转换为复数形式
        
        # 计算差分信号
        diff_signal = iq_complex[1:] * np.conj(iq_complex[:-1])
        
        # 将差分信号转换回IQ格式
        diff_i = np.real(diff_signal)
        diff_q = np.imag(diff_signal)
        
        # 创建新的信号数组，第一个样本用0填充以保持长度一致
        result = np.zeros_like(signal)
        result[1:, 0] = diff_i
        result[1:, 1] = diff_q
        
        return result

    # 所有日期
    all_dates = set(compact_dataset['capture_date_list'])
    train_dates = set(train_dates)
    test_dates = list(all_dates - train_dates)

    # 转为 list 保持顺序
    X_train, y_train = extract_samples(list(train_dates))
    X_test, y_test = extract_samples(test_dates)

    print(f"✅ 训练样本数: {len(X_train)}, 测试样本数: {len(X_test)}")
    if use_differential:
        print("✅ 已启用IQ信号差分功能")
    return X_train, y_train, X_test, y_test

def preprocess_dataset_for_classification(compact_dataset, tx_list, rx_list, train_dates, max_sig=None, equalized=0, use_phase_differential=False):
    def extract_samples(dates):
        X = []
        y = []
        sample_count = 0
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

                    for sample_idx, sample in enumerate(sig_data):
                        if sample.shape == (256, 2):
                            original_sample = sample.copy()
                            
                            # 应用相位差分
                            if use_phase_differential:
                                sample = apply_phase_differential(sample)
                            
                            # 输出第一个样本的对比信息
                            if sample_count == 0 and use_phase_differential:
                                print("=== 相位差分前后对比 ===")
                                print("差分前的前5个IQ样本值:")
                                for i in range(min(5, len(original_sample))):
                                    print(f"  样本{i}: I={original_sample[i, 0]:.6f}, Q={original_sample[i, 1]:.6f}")
                                
                                print(f"\n相位差分后的前5个样本值:")
                                for i in range(min(5, len(sample))):
                                    print(f"  样本{i}: I={sample[i, 0]:.6f}, Q={sample[i, 1]:.6f}")
                                
                                # 输出数值范围信息
                                print(f"\n数值范围信息:")
                                print(f"  原始信号I范围: [{np.min(original_sample[:, 0]):.6f}, {np.max(original_sample[:, 0]):.6f}]")
                                print(f"  原始信号Q范围: [{np.min(original_sample[:, 1]):.6f}, {np.max(original_sample[:, 1]):.6f}]")
                                print(f"  差分后I范围: [{np.min(sample[:, 0]):.6f}, {np.max(sample[:, 0]):.6f}]")
                                print(f"  差分后Q范围: [{np.min(sample[:, 1]):.6f}, {np.max(sample[:, 1]):.6f}]")
                            
                            X.append(sample)
                            y.append(tx_idx)
                            sample_count += 1
        
        return np.array(X), np.array(y)
    
    def apply_phase_differential(signal):
        """
        相位差分：保持幅度信息，消除载波频偏
        signal: 形状为 (256, 2) 的数组
        返回: 形状为 (255, 2) 的相位差分信号
        """
        iq_complex = signal[:, 0] + 1j * signal[:, 1]
        
        # 计算相位差
        phase_original = np.angle(iq_complex)
        phase_diff = np.diff(phase_original)
        
        # 将相位差包装到[-π, π]范围内
        phase_diff = np.angle(np.exp(1j * phase_diff))
        
        # 使用原始信号的幅度，结合相位差
        magnitude = np.abs(iq_complex[1:])
        diff_complex = magnitude * np.exp(1j * phase_diff)
        
        result = np.zeros((len(signal)-1, 2))
        result[:, 0] = np.real(diff_complex)
        result[:, 1] = np.imag(diff_complex)
        
        return result

    # 所有日期
    all_dates = set(compact_dataset['capture_date_list'])
    train_dates = set(train_dates)
    test_dates = list(all_dates - train_dates)

    # 转为 list 保持顺序
    X_train, y_train = extract_samples(list(train_dates))
    X_test, y_test = extract_samples(test_dates)

    print(f"✅ 训练样本数: {len(X_train)}, 测试样本数: {len(X_test)}")
    if use_phase_differential:
        print("✅ 已启用相位差分功能")
        print("✅ 相位差分：保持幅度信息，消除载波频偏")
        if len(X_train) > 0:
            print(f"✅ 差分后样本长度: {X_train.shape[1]}")
    
    return X_train, y_train, X_test, y_test

def preprocess_dataset_cross_IQ_blocks_all_mix_random(compact_dataset, tx_list, 
                                                      max_sig=None, equalized=0, 
                                                      block_size=256, y=10, 
                                                      test_ratio=0.2, seed=42):
    """
    将所有日期、所有 RX 的信号混合后随机划分训练/测试集
    每个 block 大小为 block_size，每个样本长度为 256
    """
    import numpy as np
    np.random.seed(seed)

    X = []
    y_labels = []

    for tx_idx, tx in enumerate(tx_list):
        tx_i = compact_dataset['tx_list'].index(tx)
        eq_i = compact_dataset['equalized_list'].index(equalized)

        all_tx_data = []

        # 汇总所有日期、所有 RX
        for date_i, date in enumerate(compact_dataset['capture_date_list']):
            for rx_i in range(len(compact_dataset['rx_list'])):
                sig_data = compact_dataset['data'][tx_i][rx_i][date_i][eq_i]  # shape: (N, 256, 2)
                if max_sig is not None:
                    sig_data = sig_data[:max_sig]
                all_tx_data.append(sig_data)

        if not all_tx_data:
            continue

        all_tx_data = np.concatenate(all_tx_data, axis=0)  # shape: (total_samples, 256, 2)
        np.random.shuffle(all_tx_data)  # 全局打乱

        # 将信号划分为 y 个一组
        num_chunks = len(all_tx_data) // y
        chunks = [all_tx_data[i*y:(i+1)*y] for i in range(num_chunks)]
        combined_data = np.concatenate(chunks, axis=0)

        # 分 block、转置、拆样本
        num_signals = len(combined_data)
        num_blocks = num_signals // block_size

        for i in range(num_blocks):
            block = combined_data[i*block_size:(i+1)*block_size]
            if block.shape != (block_size, 256, 2):
                continue
            block_transposed = block.transpose(1, 0, 2)  # (256, block_size, 2)
            for j in range(block_transposed.shape[0]):
                X.append(block_transposed[j])
                y_labels.append(tx_idx)

    # 转为 numpy
    X = np.array(X)
    y = np.array(y_labels)

    # 全局打乱
    perm = np.random.permutation(len(X))
    X, y = X[perm], y[perm]

    # 划分训练/测试集
    test_size = int(len(X) * test_ratio)
    X_test = X[:test_size]
    y_test = y[:test_size]
    X_train = X[test_size:]
    y_train = y[test_size:]

    print(f"✅ 总样本数: {len(X)}, 训练集: {len(X_train)}, 测试集: {len(X_test)}")
    
    return X_train, y_train, X_test, y_test


def preprocess_dataset_cross_IQ_blocks_date_interleaved_random_rx(compact_dataset, tx_list, train_dates, 
                                                                 max_sig=None, equalized=0, block_size=240, y=10):
    import numpy as np
    
    def extract_samples(dates):
        X = []
        y_labels = []

        for tx_idx, tx in enumerate(tx_list):
            tx_i = compact_dataset['tx_list'].index(tx)
            eq_i = compact_dataset['equalized_list'].index(equalized)

            # === 为每个日期和每个 RX 提取信号，并打乱 ===
            date_rx_signal_dict = {}  # date -> list of rx signals
            for date in dates:
                if date not in compact_dataset['capture_date_list']:
                    continue
                date_i = compact_dataset['capture_date_list'].index(date)
                rx_signals = []
                for rx_i in range(len(compact_dataset['rx_list'])):
                    sig_data = compact_dataset['data'][tx_i][rx_i][date_i][eq_i]
                    if max_sig is not None:
                        sig_data = sig_data[:max_sig]
                    np.random.shuffle(sig_data)
                    rx_signals.append(list(sig_data))  # 转成 list 方便 pop
                date_rx_signal_dict[date] = rx_signals

            # === 构建 block ===
            combined_data = []
            while True:
                block_chunk = []
                for date, rx_signals in date_rx_signal_dict.items():
                    for rx_idx in range(len(rx_signals)):
                        sig_list = rx_signals[rx_idx]
                        if len(sig_list) >= y:
                            # 随机抽 y 条
                            sampled = [sig_list.pop(np.random.randint(len(sig_list))) for _ in range(y)]
                            block_chunk.extend(sampled)
                        else:
                            # 如果剩余信号不足 y，就用剩下的全部
                            block_chunk.extend(sig_list)
                            rx_signals[rx_idx] = []
                if len(block_chunk) == 0:
                    break  # 没有可用信号了
                combined_data.extend(block_chunk)

            # === 分 block、转置、拆样本 ===
            num_signals = len(combined_data)
            num_blocks = num_signals // block_size
            combined_data = np.array(combined_data)

            for i in range(num_blocks):
                block = combined_data[i * block_size : (i + 1) * block_size]
                if block.shape != (block_size, 256, 2):
                    continue
                block_transposed = block.transpose(1, 0, 2)  # (256, block_size, 2)
                for j in range(block_transposed.shape[0]):
                    X.append(block_transposed[j])
                    y_labels.append(tx_idx)

        return np.array(X), np.array(y_labels)

    X_train, y_train = extract_samples(train_dates)
    test_dates = [d for d in compact_dataset['capture_date_list'] if d not in train_dates]
    X_test, y_test = extract_samples(test_dates)

    return X_train, y_train, X_test, y_test

def preprocess_dataset_cross_IQ_blocks_date_per_rx_cyclic(compact_dataset, tx_list, train_dates, 
                                                          max_sig=None, equalized=0, block_size=240, y=10):
    import numpy as np

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

                # === 提取该日期下每个 RX 的信号，并打乱 ===
                rx_signals = []
                for rx_i in range(len(compact_dataset['rx_list'])):
                    sig_data = compact_dataset['data'][tx_i][rx_i][date_i][eq_i]
                    if max_sig is not None:
                        sig_data = sig_data[:max_sig]
                    np.random.shuffle(sig_data)
                    rx_signals.append(list(sig_data))

                num_rx = len(rx_signals)
                rx_pointer = 0  # 记录当前从哪个 RX 开始抽
                accum_block = []

                while any(len(sig_list) > 0 for sig_list in rx_signals):
                    # 从 rx_pointer 开始循环抽取
                    rx_idx = rx_pointer % num_rx
                    sig_list = rx_signals[rx_idx]
                    if len(sig_list) > 0:
                        take_n = min(y, len(sig_list))
                        sampled = [sig_list.pop(0) for _ in range(take_n)]  # 顺序抽取
                        accum_block.extend(sampled)

                    rx_pointer += 1

                    # 当累积到 block_size 时生成 block
                    while len(accum_block) >= block_size:
                        block_chunk = accum_block[:block_size]
                        accum_block = accum_block[block_size:]
                        block_array = np.array(block_chunk)  # (block_size, 256, 2)
                        block_transposed = block_array.transpose(1, 0, 2)  # (256, block_size, 2)
                        for j in range(block_transposed.shape[0]):
                            X.append(block_transposed[j])
                            y_labels.append(tx_idx)

                # === 如果最后剩余的 accum_block 不足 block_size，这里选择丢弃 ===
                accum_block = []

        return np.array(X), np.array(y_labels)

    X_train, y_train = extract_samples(train_dates)
    test_dates = [d for d in compact_dataset['capture_date_list'] if d not in train_dates]
    X_test, y_test = extract_samples(test_dates)

    return X_train, y_train, X_test, y_test

def preprocess_dataset_for_classification_random_split(compact_dataset, tx_list, rx_list, 
                                                        test_ratio=0.25, max_sig=None, 
                                                        equalized=0, use_phase_differential=False, seed=42):
    import numpy as np

    def extract_all_samples():
        X = []
        y = []
        sample_count = 0
        rng = np.random.default_rng(seed)

        for rx in rx_list:
            for tx_idx, tx in enumerate(tx_list):
                tx_i = compact_dataset['tx_list'].index(tx)
                rx_i = compact_dataset['rx_list'].index(rx)
                eq_i = compact_dataset['equalized_list'].index(equalized)

                for date_i, date in enumerate(compact_dataset['capture_date_list']):
                    sig_data = compact_dataset['data'][tx_i][rx_i][date_i][eq_i]
                    if max_sig is not None:
                        sig_data = sig_data[:max_sig]

                    for sample_idx, sample in enumerate(sig_data):
                        if sample.shape == (256, 2):
                            original_sample = sample.copy()

                            if use_phase_differential:
                                sample = apply_phase_differential(sample)

                                # 输出第一个样本的对比信息
                                if sample_count == 0:
                                    print("=== 相位差分前后对比 ===")
                                    print("差分前的前5个IQ样本值:")
                                    for i in range(min(5, len(original_sample))):
                                        print(f"  样本{i}: I={original_sample[i, 0]:.6f}, Q={original_sample[i, 1]:.6f}")
                                    print(f"\n相位差分后的前5个样本值:")
                                    for i in range(min(5, len(sample))):
                                        print(f"  样本{i}: I={sample[i, 0]:.6f}, Q={sample[i, 1]:.6f}")
                                    print(f"\n数值范围信息:")
                                    print(f"  原始信号I范围: [{np.min(original_sample[:, 0]):.6f}, {np.max(original_sample[:, 0]):.6f}]")
                                    print(f"  原始信号Q范围: [{np.min(original_sample[:, 1]):.6f}, {np.max(original_sample[:, 1]):.6f}]")
                                    print(f"  差分后I范围: [{np.min(sample[:, 0]):.6f}, {np.max(sample[:, 0]):.6f}]")
                                    print(f"  差分后Q范围: [{np.min(sample[:, 1]):.6f}, {np.max(sample[:, 1]):.6f}]")
                            
                            X.append(sample)
                            y.append(tx_idx)
                            sample_count += 1
        return np.array(X), np.array(y)

    def apply_phase_differential(signal):
        iq_complex = signal[:, 0] + 1j * signal[:, 1]
        phase_original = np.angle(iq_complex)
        phase_diff = np.diff(phase_original)
        phase_diff = np.angle(np.exp(1j * phase_diff))
        magnitude = np.abs(iq_complex[1:])
        diff_complex = magnitude * np.exp(1j * phase_diff)
        result = np.zeros((len(signal)-1, 2))
        result[:, 0] = np.real(diff_complex)
        result[:, 1] = np.imag(diff_complex)
        return result

    # === 提取所有样本 ===
    X_all, y_all = extract_all_samples()
    print(f"✅ 总样本数: {len(X_all)}")

    # === 随机划分训练/测试集 ===
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(X_all))
    test_size = int(len(X_all) * test_ratio)
    test_idx = indices[:test_size]
    train_idx = indices[test_size:]

    X_train = X_all[train_idx]
    y_train = y_all[train_idx]
    X_test = X_all[test_idx]
    y_test = y_all[test_idx]

    print(f"✅ 训练样本数: {len(X_train)}, 测试样本数: {len(X_test)}")
    if use_phase_differential:
        print("✅ 已启用相位差分功能")
        print(f"✅ 差分后样本长度: {X_train.shape[1]}")

    return X_train, y_train, X_test, y_test

def preprocess_dataset_cross_IQ_blocks_single_date_per_rx_ran(compact_dataset, tx_list, train_dates, test_dates,
                                                          max_sig=None, equalized=0, block_size=240, y=10):
    import numpy as np

    def extract_samples(dates):
        X = []
        y_labels = []

        for tx_idx, tx in enumerate(tx_list):
            try:
                tx_i = compact_dataset['tx_list'].index(tx)
            except ValueError:
                continue  # 如果 tx 不在列表中，跳过
            try:
                eq_i = compact_dataset['equalized_list'].index(equalized)
            except ValueError:
                continue

            for date in dates:
                if date not in compact_dataset['capture_date_list']:
                    continue
                date_i = compact_dataset['capture_date_list'].index(date)

                # === 提取该日期下每个 RX 的信号，并打乱 ===
                rx_signals = []
                for rx_i in range(len(compact_dataset['rx_list'])):
                    sig_data = compact_dataset['data'][tx_i][rx_i][date_i][eq_i]
                    if max_sig is not None:
                        sig_data = sig_data[:max_sig]
                    sig_data = list(sig_data)
                    np.random.shuffle(sig_data)
                    rx_signals.append(sig_data)

                num_rx = len(rx_signals)
                rx_pointer = 0  # 当前从哪个 RX 开始抽
                accum_block = []

                while any(len(sig_list) > 0 for sig_list in rx_signals):
                    rx_idx = rx_pointer % num_rx
                    sig_list = rx_signals[rx_idx]
                    if len(sig_list) > 0:
                        take_n = min(y, len(sig_list))
                        sampled = [sig_list.pop(0) for _ in range(take_n)]
                        accum_block.extend(sampled)
                    rx_pointer += 1

                    # 当累积到 block_size 时生成 block
                    while len(accum_block) >= block_size:
                        block_chunk = accum_block[:block_size]
                        accum_block = accum_block[block_size:]
                        block_array = np.array(block_chunk)  # (block_size, 256, 2)
                        block_transposed = block_array.transpose(1, 0, 2)  # (256, block_size, 2)
                        for j in range(block_transposed.shape[0]):
                            X.append(block_transposed[j])
                            y_labels.append(tx_idx)

                # 最后剩余的 accum_block 不足 block_size 时丢弃
                accum_block = []

        return np.array(X), np.array(y_labels)

    # 训练集
    X_train, y_train = extract_samples(train_dates)

    # 测试集
    X_test, y_test = extract_samples(test_dates)

    return X_train, y_train, X_test, y_test

def preprocess_dataset_for_classification_cross_date(compact_dataset, tx_list, rx_list, train_dates, test_dates, max_sig=None, equalized=0, use_phase_differential=False):
    def extract_samples(dates):
        X = []
        y = []
        sample_count = 0
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

                    for sample_idx, sample in enumerate(sig_data):
                        if sample.shape == (256, 2):
                            original_sample = sample.copy()
                            
                            # 应用相位差分
                            if use_phase_differential:
                                sample = apply_phase_differential(sample)
                            
                            # 输出第一个样本的对比信息
                            if sample_count == 0 and use_phase_differential:
                                print("=== 相位差分前后对比 ===")
                                print("差分前的前5个IQ样本值:")
                                for i in range(min(5, len(original_sample))):
                                    print(f"  样本{i}: I={original_sample[i, 0]:.6f}, Q={original_sample[i, 1]:.6f}")
                                
                                print(f"\n相位差分后的前5个样本值:")
                                for i in range(min(5, len(sample))):
                                    print(f"  样本{i}: I={sample[i, 0]:.6f}, Q={sample[i, 1]:.6f}")
                                
                                # 输出数值范围信息
                                print(f"\n数值范围信息:")
                                print(f"  原始信号I范围: [{np.min(original_sample[:, 0]):.6f}, {np.max(original_sample[:, 0]):.6f}]")
                                print(f"  原始信号Q范围: [{np.min(original_sample[:, 1]):.6f}, {np.max(original_sample[:, 1]):.6f}]")
                                print(f"  差分后I范围: [{np.min(sample[:, 0]):.6f}, {np.max(sample[:, 0]):.6f}]")
                                print(f"  差分后Q范围: [{np.min(sample[:, 1]):.6f}, {np.max(sample[:, 1]):.6f}]")
                            
                            X.append(sample)
                            y.append(tx_idx)
                            sample_count += 1
        
        return np.array(X), np.array(y)
    
    def apply_phase_differential(signal):
        """
        相位差分：保持幅度信息，消除载波频偏
        signal: 形状为 (256, 2) 的数组
        返回: 形状为 (255, 2) 的相位差分信号
        """
        iq_complex = signal[:, 0] + 1j * signal[:, 1]
        
        # 计算相位差
        phase_original = np.angle(iq_complex)
        phase_diff = np.diff(phase_original)
        
        # 将相位差包装到[-π, π]范围内
        phase_diff = np.angle(np.exp(1j * phase_diff))
        
        # 使用原始信号的幅度，结合相位差
        magnitude = np.abs(iq_complex[1:])
        diff_complex = magnitude * np.exp(1j * phase_diff)
        
        result = np.zeros((len(signal)-1, 2))
        result[:, 0] = np.real(diff_complex)
        result[:, 1] = np.imag(diff_complex)
        
        return result

    # 所有日期
    all_dates = set(compact_dataset['capture_date_list'])
    train_dates = set(train_dates)
    test_dates = set(test_dates)

    # 转为 list 保持顺序
    X_train, y_train = extract_samples(list(train_dates))
    X_test, y_test = extract_samples(list(test_dates))

    print(f"✅ 训练样本数: {len(X_train)}, 测试样本数: {len(X_test)}")
    if use_phase_differential:
        print("✅ 已启用相位差分功能")
        print("✅ 相位差分：保持幅度信息，消除载波频偏")
        if len(X_train) > 0:
            print(f"✅ 差分后样本长度: {X_train.shape[1]}")
    
    return X_train, y_train, X_test, y_test

# ------------------------ 数据集处理函数（block-level，严格 TX-日期-RX 顺序） ------------------------
def preprocess_dataset_cross_IQ_independent_blocks_per_tx_day(compact_dataset, tx_list, 
                                                              max_sig=None, equalized=0, 
                                                              block_size=256, y=10, 
                                                              test_ratio=0.2, seed=42):
    """
    返回训练集 block、训练标签 block、测试集 block、测试标签 block

    每个 block 严格按照：
    TX -> 日期 -> RX 顺序抽取，每个 RX 抽 y 条信号循环填充 block_size
    block shape: (256, block_size, 2)

    核心改进：
    - 支持跨多轮 RX 循环累积信号填满 block_size
    - 不会出现空训练集
    """
    import numpy as np
    np.random.seed(seed)

    train_blocks, train_block_labels = [], []
    test_blocks, test_block_labels = [], []

    # 遍历每个 TX
    for tx_idx, tx in enumerate(tx_list):
        tx_i = compact_dataset['tx_list'].index(tx)
        eq_i = compact_dataset['equalized_list'].index(equalized)

        # 遍历每个日期
        for date_i, date in enumerate(compact_dataset['capture_date_list']):
            # 收集当前 TX 当前日期下所有 RX 的信号
            rx_signals = []
            for rx_i in range(len(compact_dataset['rx_list'])):
                sig_data = compact_dataset['data'][tx_i][rx_i][date_i][eq_i]  # shape: (N_sig, 256, 2)
                if max_sig is not None:
                    sig_data = sig_data[:max_sig]
                rx_signals.append(sig_data)

            # 每个 RX 的指针，表示已经抽取的信号位置
            rx_pointers = [0] * len(rx_signals)
            rx_lengths = [len(arr) for arr in rx_signals]

            # 循环生成 block，直到所有 RX 信号都抽完
            while True:
                block_list = []  # 存放本次 block 累积的信号
                # 标记是否还有信号可以抽
                any_signal_left = False

                # 循环 RX，直到 block 达到 block_size
                while sum([b.shape[0] for b in block_list]) < block_size:
                    finished = True  # 假设所有 RX 都抽完
                    for rx_idx, arr in enumerate(rx_signals):
                        start = rx_pointers[rx_idx]
                        end = min(start + y, rx_lengths[rx_idx])
                        if start >= rx_lengths[rx_idx]:
                            continue  # 当前 RX 没有剩余信号
                        finished = False
                        any_signal_left = True
                        block_list.append(arr[start:end])
                        rx_pointers[rx_idx] = end

                        # 如果 block 已经达到 block_size，提前停止循环
                        if sum([b.shape[0] for b in block_list]) >= block_size:
                            break
                    if finished:
                        break  # 所有 RX 都抽完信号

                if not any_signal_left:
                    # 所有 RX 信号都用完，退出循环
                    break

                # 拼接 block
                block_array = np.concatenate(block_list, axis=0)

                # 截取到 block_size
                if block_array.shape[0] >= block_size:
                    block_array = block_array[:block_size]
                else:
                    # 理论上不会走到这里，但保险起见跳过不足 block_size 的 block
                    continue

                # 转置 block: (block_size, 256, 2) -> (256, block_size, 2)
                block_transposed = block_array.transpose(1, 0, 2)

                # 随机分配到训练/测试
                if np.random.rand() < test_ratio:
                    test_blocks.append(block_transposed)
                    test_block_labels.append(tx_idx)
                else:
                    train_blocks.append(block_transposed)
                    train_block_labels.append(tx_idx)

    # 转成 numpy array
    train_blocks = np.array(train_blocks)   # [N_block, 256, block_size, 2]
    train_block_labels = np.array(train_block_labels)
    test_blocks = np.array(test_blocks)
    test_block_labels = np.array(test_block_labels)

    print(f"✅ 总 block 数: {len(train_blocks)+len(test_blocks)}, "
          f"训练集 block: {len(train_blocks)}, 测试集 block: {len(test_blocks)}")
    if len(train_blocks) > 0:
        print(f"✅ 每个 block shape: {train_blocks[0].shape}")
    return train_blocks, train_block_labels, test_blocks, test_block_labels

def preprocess_dataset_cross_IQ_blocks_single_date_per_rx_cyclic(
        compact_dataset, tx_list, train_dates, test_dates,
        max_sig=None, equalized=0, block_size=240, y=10):

    import numpy as np

    def extract_samples(dates):
        """
        对一组日期进行处理：
        - 每个 TX
        - 每个日期
        - 循环遍历所有 RX
        - 从每个 RX 顺序抽取 y 条信号
        - 持续累积，直到构成 block_size
        - 按 block_size 切块为 block
        - 每个 block 再按 (block_size, 256, 2) → (256, block_size, 2)
        """

        X = []
        y_labels = []

        for tx_idx, tx in enumerate(tx_list):
            # === 找到 TX 索引 ===
            try:
                tx_i = compact_dataset["tx_list"].index(tx)
            except ValueError:
                continue

            # === EQ 索引 ===
            try:
                eq_i = compact_dataset['equalized_list'].index(equalized)
            except ValueError:
                continue

            # ================
            #  遍历每一个日期
            # ================
            for date in dates:
                if date not in compact_dataset['capture_date_list']:
                    continue
                date_i = compact_dataset['capture_date_list'].index(date)

                # =======================================================
                #  (1) 获取每个 RX 的信号序列（顺序，不打乱）
                # =======================================================
                rx_signals = []
                for rx_i in range(len(compact_dataset['rx_list'])):

                    sig_data = compact_dataset['data'][tx_i][rx_i][date_i][eq_i]

                    # 限制最多使用多少条
                    if max_sig is not None:
                        sig_data = sig_data[:max_sig]

                    # 转成 list 便于 pop(0)
                    sig_data = list(sig_data)

                    # 不打乱，保持原始顺序
                    rx_signals.append(sig_data)

                num_rx = len(rx_signals)
                rx_pointer = 0   # 用于轮询 RX
                accum_block = [] # 当前 block 的累积信号

                # =======================================================
                #  (2) 循环抽取：轮询 RX → 从每个 RX 取 y 条顺序信号
                # =======================================================
                while any(len(sig_list) > 0 for sig_list in rx_signals):

                    rx_idx = rx_pointer % num_rx
                    sig_list = rx_signals[rx_idx]

                    if len(sig_list) > 0:
                        # 从该 RX 中取 y 条顺序信号
                        take_n = min(y, len(sig_list))
                        sampled = [sig_list.pop(0) for _ in range(take_n)]

                        # 放入累计区
                        accum_block.extend(sampled)

                    rx_pointer += 1

                    # =======================================================
                    # (3) 如果 accum_block >= block_size，则生成 block
                    # =======================================================
                    while len(accum_block) >= block_size:
                        block_chunk = accum_block[:block_size]
                        accum_block = accum_block[block_size:]

                        block_array = np.array(block_chunk)          # (block_size, 256, 2)
                        block_transposed = block_array.transpose(1, 0, 2)  # (256, block_size, 2)

                        # 这里每个 block 对应一个 TX
                        for k in range(block_transposed.shape[0]):
                            X.append(block_transposed[k])
                            y_labels.append(tx_idx)

                # =======================================================
                # (4) 当前日期抽取剩余不足 block_size 的部分丢弃
                # =======================================================
                accum_block = []

        return np.array(X), np.array(y_labels)

    # =====================================
    # 生成训练集
    # =====================================
    X_train, y_train = extract_samples(train_dates)

    # =====================================
    # 生成测试集
    # =====================================
    X_test, y_test = extract_samples(test_dates)

    return X_train, y_train, X_test, y_test

def preprocess_dataset_for_classification_tx_split(
    compact_dataset, 
    tx_list, 
    rx_list,
    test_ratio=0.25, 
    max_sig=None, 
    equalized=0, 
    use_phase_differential=False, 
    seed=42,
):
    """
    按 TX 划分训练/测试集。每个 TX 下的所有 RX、所有日期的样本，
    都按 test_ratio 进行独立拆分，避免 TX 样本不均衡问题。

    参数说明：
    ------------
    compact_dataset: 压缩数据集结构
    tx_list: 需要使用的 TX 列表
    rx_list: 需要使用的 RX 列表
    test_ratio: 测试集比例（对每个 TX 独立划分）
    max_sig: 每条 RX 最多使用的样本数
    equalized: EQ 模式索引
    use_phase_differential: 是否启用相位差分
    seed: 随机种子

    返回：
    ------------
    X_train, y_train, X_test, y_test
    """

    import numpy as np
    rng = np.random.default_rng(seed)

    def apply_phase_differential(signal):
        """对单条 IQ 信号进行相位差分"""
        iq_complex = signal[:, 0] + 1j * signal[:, 1]
        phase_original = np.angle(iq_complex)
        phase_diff = np.diff(phase_original)
        # 使用 wrap 方式
        phase_diff = np.angle(np.exp(1j * phase_diff))
        magnitude = np.abs(iq_complex[1:])
        diff_complex = magnitude * np.exp(1j * phase_diff)

        # 返回 (L-1, 2)
        out = np.zeros((len(signal) - 1, 2))
        out[:, 0] = diff_complex.real
        out[:, 1] = diff_complex.imag
        return out

    # ----------------------------
    # 最终输出的容器
    # ----------------------------
    X_train, y_train = [], []
    X_test,  y_test  = [], []

    print("==============================================")
    print("开始按 TX 划分训练/测试集 ...")
    print(f"TX 数量：{len(tx_list)}, RX 数量：{len(rx_list)}, 日期数：{len(compact_dataset['capture_date_list'])}")
    print("==============================================")

    # -------------------------------------------------
    # 逐个 TX 处理 → 每个 TX 独立划分比例
    # -------------------------------------------------
    for tx_idx, tx in enumerate(tx_list):

        print(f"\n▶ 处理 TX = {tx} (index={tx_idx})")

        # 该 TX 的所有样本会暂存到临时 buffer
        tx_samples = []

        # 找到 TX、EQ 的索引
        try:
            tx_i = compact_dataset['tx_list'].index(tx)
        except ValueError:
            print(f"⚠ TX {tx} 不在数据集中，跳过")
            continue

        try:
            eq_i = compact_dataset['equalized_list'].index(equalized)
        except:
            print("⚠ equalized index 不存在")
            continue

        # -------------------------------------------------
        # 遍历所有 RX、所有日期，提取该 TX 下的全部信号
        # -------------------------------------------------
        for rx in rx_list:
            try:
                rx_i = compact_dataset['rx_list'].index(rx)
            except:
                continue

            for date_i, date in enumerate(compact_dataset['capture_date_list']):

                sig_data = compact_dataset['data'][tx_i][rx_i][date_i][eq_i]

                if max_sig is not None:
                    sig_data = sig_data[:max_sig]

                for sig in sig_data:  # shape (256,2)
                    sample = sig.copy()

                    if use_phase_differential:
                        sample = apply_phase_differential(sample)

                    # 收集该 TX 的所有样本
                    tx_samples.append(sample)

        tx_samples = np.array(tx_samples)  # (N_tx, L, 2)
        N_tx = len(tx_samples)
        print(f"    TX {tx} 总样本数: {N_tx}")

        # -------------------------------------------------
        # 对该 TX 单独随机划分训练/测试
        # -------------------------------------------------
        indices = rng.permutation(N_tx)
        test_size = int(N_tx * test_ratio)
        test_idx_tx = indices[:test_size]
        train_idx_tx = indices[test_size:]

        # 存入全局
        for i in train_idx_tx:
            X_train.append(tx_samples[i])
            y_train.append(tx_idx)

        for i in test_idx_tx:
            X_test.append(tx_samples[i])
            y_test.append(tx_idx)

        print(f"    → 训练: {len(train_idx_tx)} 条, 测试: {len(test_idx_tx)} 条")

    # ----------------------------
    # 转为 np.array 输出
    # ----------------------------
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test  = np.array(X_test)
    y_test  = np.array(y_test)

    print("\n==============================================")
    print("最终数据统计")
    print(f"训练集: X={X_train.shape}, y={y_train.shape}")
    print(f"测试集: X={X_test.shape},  y={y_test.shape}")
    print("==============================================")

    if use_phase_differential:
        print("✔ 已启用相位差分，样本长度变为:", X_train.shape[1])

    return X_train, y_train, X_test, y_test

def preprocess_dataset_for_classification_tx_split(
    compact_dataset, 
    tx_list, 
    rx_list,
    test_ratio=0.25, 
    max_sig=None, 
    equalized=0, 
    use_phase_differential=False, 
    seed=42,
):
    """
    按 TX 划分训练/测试集。每个 TX 下的所有 RX、所有日期的样本，
    都按 test_ratio 进行独立拆分，避免 TX 样本不均衡问题。

    参数说明：
    ------------
    compact_dataset: 压缩数据集结构
    tx_list: 需要使用的 TX 列表
    rx_list: 需要使用的 RX 列表
    test_ratio: 测试集比例（对每个 TX 独立划分）
    max_sig: 每条 RX 最多使用的样本数
    equalized: EQ 模式索引
    use_phase_differential: 是否启用相位差分
    seed: 随机种子

    返回：
    ------------
    X_train, y_train, X_test, y_test
    """

    import numpy as np
    rng = np.random.default_rng(seed)

    def apply_phase_differential(signal):
        """对单条 IQ 信号进行相位差分"""
        iq_complex = signal[:, 0] + 1j * signal[:, 1]
        phase_original = np.angle(iq_complex)
        phase_diff = np.diff(phase_original)
        # 使用 wrap 方式
        phase_diff = np.angle(np.exp(1j * phase_diff))
        magnitude = np.abs(iq_complex[1:])
        diff_complex = magnitude * np.exp(1j * phase_diff)

        # 返回 (L-1, 2)
        out = np.zeros((len(signal) - 1, 2))
        out[:, 0] = diff_complex.real
        out[:, 1] = diff_complex.imag
        return out

    # ----------------------------
    # 最终输出的容器
    # ----------------------------
    X_train, y_train = [], []
    X_test,  y_test  = [], []

    print("==============================================")
    print("开始按 TX 划分训练/测试集 ...")
    print(f"TX 数量：{len(tx_list)}, RX 数量：{len(rx_list)}, 日期数：{len(compact_dataset['capture_date_list'])}")
    print("==============================================")

    # -------------------------------------------------
    # 逐个 TX 处理 → 每个 TX 独立划分比例
    # -------------------------------------------------
    for tx_idx, tx in enumerate(tx_list):

        print(f"\n▶ 处理 TX = {tx} (index={tx_idx})")

        # 该 TX 的所有样本会暂存到临时 buffer
        tx_samples = []

        # 找到 TX、EQ 的索引
        try:
            tx_i = compact_dataset['tx_list'].index(tx)
        except ValueError:
            print(f"⚠ TX {tx} 不在数据集中，跳过")
            continue

        try:
            eq_i = compact_dataset['equalized_list'].index(equalized)
        except:
            print("⚠ equalized index 不存在")
            continue

        # -------------------------------------------------
        # 遍历所有 RX、所有日期，提取该 TX 下的全部信号
        # -------------------------------------------------
        for rx in rx_list:
            try:
                rx_i = compact_dataset['rx_list'].index(rx)
            except:
                continue

            for date_i, date in enumerate(compact_dataset['capture_date_list']):

                sig_data = compact_dataset['data'][tx_i][rx_i][date_i][eq_i]

                if max_sig is not None:
                    sig_data = sig_data[:max_sig]

                for sig in sig_data:  # shape (256,2)
                    sample = sig.copy()

                    if use_phase_differential:
                        sample = apply_phase_differential(sample)

                    # 收集该 TX 的所有样本
                    tx_samples.append(sample)

        tx_samples = np.array(tx_samples)  # (N_tx, L, 2)
        N_tx = len(tx_samples)
        print(f"    TX {tx} 总样本数: {N_tx}")

        # -------------------------------------------------
        # 对该 TX 单独随机划分训练/测试
        # -------------------------------------------------
        indices = rng.permutation(N_tx)
        test_size = int(N_tx * test_ratio)
        test_idx_tx = indices[:test_size]
        train_idx_tx = indices[test_size:]

        # 存入全局
        for i in train_idx_tx:
            X_train.append(tx_samples[i])
            y_train.append(tx_idx)

        for i in test_idx_tx:
            X_test.append(tx_samples[i])
            y_test.append(tx_idx)

        print(f"    → 训练: {len(train_idx_tx)} 条, 测试: {len(test_idx_tx)} 条")

    # ----------------------------
    # 转为 np.array 输出
    # ----------------------------
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test  = np.array(X_test)
    y_test  = np.array(y_test)

    print("\n==============================================")
    print("最终数据统计")
    print(f"训练集: X={X_train.shape}, y={y_train.shape}")
    print(f"测试集: X={X_test.shape},  y={y_test.shape}")
    print("==============================================")

    if use_phase_differential:
        print("✔ 已启用相位差分，样本长度变为:", X_train.shape[1])

    return X_train, y_train, X_test, y_test

def preprocess_dataset_cross_IQ_balanced_blocks_per_tx_day(
        compact_dataset, tx_list, max_sig=None, equalized=0,
        block_size=180, y=10, test_ratio=0.25, seed=42):
    import numpy as np
    np.random.seed(seed)

    train_blocks, train_block_labels = [], []
    test_blocks, test_block_labels = [], []

    for tx_idx, tx in enumerate(tx_list):
        tx_i = compact_dataset['tx_list'].index(tx)
        eq_i = compact_dataset['equalized_list'].index(equalized)

        for date_i, date in enumerate(compact_dataset['capture_date_list']):
            rx_signals = []
            rx_lengths = []
            for rx_i in range(len(compact_dataset['rx_list'])):
                sig_data = compact_dataset['data'][tx_i][rx_i][date_i][eq_i]
                if max_sig is not None:
                    sig_data = sig_data[:max_sig]
                if len(sig_data) > 0:
                    rx_signals.append(sig_data)
                    rx_lengths.append(len(sig_data))

            if len(rx_signals) == 0:
                continue

            # --------- 检查：只保留长度相同的 RX，否则跳过该日期 ---------
            if len(set(rx_lengths)) > 1:
                continue

            # 初始化 RX 指针
            rx_ptrs = [0] * len(rx_signals)
            all_blocks_for_tx_date = []

            # 生成当前 TX+日期的所有 block
            while True:
                block_list = []
                any_left = False
                while sum([b.shape[0] for b in block_list]) < block_size:
                    finished = True
                    for i, arr in enumerate(rx_signals):
                        start = rx_ptrs[i]
                        end = min(start + y, len(arr))
                        if start >= len(arr):
                            continue
                        finished = False
                        any_left = True
                        block_list.append(arr[start:end])
                        rx_ptrs[i] = end
                        if sum([b.shape[0] for b in block_list]) >= block_size:
                            break
                    if finished:
                        break
                if not any_left:
                    break

                block_array = np.concatenate(block_list, axis=0)
                if block_array.shape[0] < block_size:
                    continue
                block_array = block_array[:block_size]
                block_transposed = block_array.transpose(1, 0, 2)
                all_blocks_for_tx_date.append(block_transposed)

            # --------- 严格按比例划分 train/test ---------
            n_blocks = len(all_blocks_for_tx_date)
            if n_blocks == 0:
                continue
            n_test = int(n_blocks * test_ratio)
            n_train = n_blocks - n_test

            test_blocks.extend(all_blocks_for_tx_date[:n_test])
            test_block_labels.extend([tx_idx] * n_test)
            train_blocks.extend(all_blocks_for_tx_date[n_test:])
            train_block_labels.extend([tx_idx] * n_train)

    train_blocks = np.array(train_blocks)
    train_block_labels = np.array(train_block_labels)
    test_blocks = np.array(test_blocks)
    test_block_labels = np.array(test_block_labels)

    print(f"✅ 总 block 数: {len(train_blocks)+len(test_blocks)}, "
          f"训练集 block: {len(train_blocks)}, 测试集 block: {len(test_blocks)}")
    if len(train_blocks) > 0:
        print(f"✅ 每个 block shape: {train_blocks[0].shape}")

    return train_blocks, train_block_labels, test_blocks, test_block_labels


