%% 参数配置
input_folder = '..\los_data';        % 输入数据文件夹
output_root = 'TSNE';             % 输出根目录
enable_noise = true;                 % 噪声使能开关
SNR_dB = 5;                         % 信噪比设置
group_size = 320;                    % 每组样本量
tsne_perplexity = 30;                % t-SNE困惑度参数
resolution = 300;                    % 输出图像DPI
random_seed = 2023;                  % 固定随机种子
num_selected_devices = 10;            % 随机选择的设备数量

%% 初始化环境
clc; close all; 
rng(random_seed, 'twister');

%% 获取设备列表并随机选择
mat_files = dir(fullfile(input_folder, '*.mat'));
num_devices = length(mat_files);
fprintf('发现%d个设备数据\n', num_devices);

% 校验设备选择数量
if num_selected_devices > num_devices
    error('错误：选择数量(%d)超过总设备数(%d)', num_selected_devices, num_devices);
end
selected_indices = randperm(num_devices, num_selected_devices);

%% 数据预处理管道
[feature_matrix, device_labels] = deal([]);

for d = selected_indices
    % 数据加载
    [~, dev_name] = fileparts(mat_files(d).name);
    try
        data = load(fullfile(input_folder, mat_files(d).name));
        raw_signals = data.data_Ineed.';
    catch
        fprintf('[%s] 数据加载失败\n', dev_name);
        continue;
    end

    %% 严格随机抽样
    if size(raw_signals, 1) < group_size
        fprintf('[%s] 前导码不足: %d < %d\n', dev_name, size(raw_signals,1), group_size);
        continue;
    end
    
    rand_idx = randperm(size(raw_signals, 1), group_size);
    [processed_signals, clean_idx] = data_cleaning_pipeline(raw_signals(rand_idx,:), enable_noise, SNR_dB);
    [features, valid_features] = feature_extraction_with_validation(processed_signals);
    
    if ~isempty(features)
        feature_matrix = [feature_matrix; features];
        device_labels = [device_labels; repmat({dev_name}, size(features,1), 1)];
        fprintf('[%s] 有效样本: %d/%d\n', dev_name, size(features,1), target_length);
    end
end

%% 数据完整性检查
assert(~isempty(feature_matrix), '错误：所有选中设备均无有效数据！');
actual_devices = length(unique(device_labels));

%% 降维分析
projection_2d = tsne(feature_matrix, 'NumDimensions', 2, 'Perplexity', tsne_perplexity);
projection_3d = tsne(feature_matrix, 'NumDimensions', 3, 'Perplexity', tsne_perplexity);

%% 可视化输出
visualization_engine(output_root, resolution, device_labels, projection_2d, projection_3d,...
    SNR_dB, num_selected_devices, actual_devices, random_seed);

%% 数据清洗管道
function [processed, valid_idx] = data_cleaning_pipeline(signals, noise_flag, snr)
    valid_idx = find(~all(signals == 0, 2));
    active_signals = signals(valid_idx, :);
    
    processed = active_signals;
    if noise_flag
        for i = 1:size(active_signals, 1)
            processed(i,:) = awgn(active_signals(i,:), snr, 'measured');
        end
    end
    
    nan_mask = any(isnan(processed), 2);
    processed(nan_mask,:) = [];
    valid_idx(nan_mask) = [];
end

%% 特征提取
function [features, valid_mask] = feature_extraction_with_validation(signals)
    time_features = [real(signals), imag(signals)];
    
    try
        freq_signals = fft(signals, [], 2);
        freq_features = [abs(freq_signals), angle(freq_signals)];
    catch
        freq_features = [];
    end
    
    features = [time_features, freq_features];
    valid_mask = ~any(isnan(features) | isinf(features), 2);
    features = features(valid_mask,:);
end

%% 可视化引擎
function visualization_engine(output_root, dpi, labels, proj2d, proj3d, snr, num_selected, actual_num, seed)
    file_prefix = sprintf('SNR%d_Sel%d_Act%d_Seed%d', snr, num_selected, actual_num, seed);
    
    viz_dir = fullfile(output_root, 'TSNE_Pos');
    if ~exist(viz_dir, 'dir')
        mkdir(viz_dir);
    end

    title_str = @(dim) sprintf('长时序轨迹图-IQ信号t-SNE %dD投影\nSNR: %ddB | 选择/有效设备: %d/%d | 随机种子: %d',...
        dim, snr, num_selected, actual_num, seed);

    % 2D可视化
    fig = figure('Position', [100 100 800 600], 'Visible', 'off');
    gscatter(proj2d(:,1), proj2d(:,2), grp2idx(labels), lines(length(unique(labels))), '.', 15);
    title(title_str(2));
    exportgraphics(fig, fullfile(viz_dir, [file_prefix '_2D.png']), 'Resolution', dpi);

    % 3D可视化
    fig = figure('Position', [100 100 800 600], 'Visible', 'off');
    scatter3(proj3d(:,1), proj3d(:,2), proj3d(:,3), 10, grp2idx(labels), 'filled');
    title(title_str(3));
    view(135, 30);
    grid on;
    exportgraphics(fig, fullfile(viz_dir, [file_prefix '_3D.png']), 'Resolution', dpi);
    close all;
end