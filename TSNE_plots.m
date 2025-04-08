%% 参数配置
input_folder = '..\los_data';
output_root = '..\TSNE';
enable_noise = true;
SNR_dB = 5;
target_length = 320;
tsne_perplexity = 30;
resolution = 300;
fixed_seed = 2023;  
num_selected_devices = 10;  % +++ 新增随机选择参数 +++

%% 初始化环境
clc; close all; 
rng(fixed_seed, 'twister');
mkdir(output_root);

%% 增强型数据管道（新增设备选择逻辑）
mat_files = dir(fullfile(input_folder, '*.mat'));
num_devices = length(mat_files);

% 设备选择校验
if num_selected_devices > num_devices
    error('设备选择数量错误: %d > %d', num_selected_devices, num_devices);
end
selected_idx = randperm(num_devices, num_selected_devices);  % +++ 核心选择逻辑 +++

[feature_matrix, device_labels] = deal([]);
for d = selected_idx  % +++ 仅处理选中设备 +++
    [~, dev_name] = fileparts(mat_files(d).name);
    try
        % 数据加载与校验
        data = load(fullfile(input_folder, mat_files(d).name));
        raw_data = data.data_Ineed;
        
        % 随机抽样（保持原逻辑）
        num_signals = size(raw_data, 2);
        if num_signals < target_length
            fprintf('[%s] 信号不足: %d < %d\n', dev_name, num_signals, target_length);
            continue;
        end
        rand_idx = randperm(num_signals, target_length);
        selected_data = raw_data(:, rand_idx);
        
    catch
        fprintf('[%s] 数据加载失败\n', dev_name);
        continue; 
    end
    
    % 信号预处理
    valid_signals = process_iq_signals(selected_data, target_length, enable_noise, SNR_dB);
    
    % 特征提取
    [features, valid_idx] = extract_tsne_features(valid_signals);
    
    % 数据收集（新增有效设备统计）
    if ~isempty(features)
        feature_matrix = [feature_matrix; features];
        device_labels = [device_labels; repmat({dev_name}, size(features,1), 1)];
        fprintf('[%s] 有效样本: %d/%d\n', dev_name, size(features,1), target_length);
    end
end

%% 增强型数据检查（新增设备有效性验证）
assert(~isempty(feature_matrix), '所有选中设备均无有效数据！');
actual_devices = length(unique(device_labels));

%% 可视化（新增参数传递）
visualize_tsne_results(feature_matrix, device_labels, output_root, resolution, tsne_perplexity,...
    SNR_DB=SNR_dB, SelectedDevices=num_selected_devices,...
    ActualDevices=actual_devices, Seed=fixed_seed);  % +++ 新增命名参数传递 +++

%% 信号处理函数（保持原样）
function valid_signals = process_iq_signals(raw_data, target_len, enable_noise, snr)
    valid_signals = [];
    if isempty(raw_data)
        return;
    end
    
    processed = zeros(target_len, size(raw_data,2));
    for col = 1:size(raw_data,2)
        sig = raw_data(1:min(end,target_len), col);
        if length(sig) < target_len
            sig = [sig; zeros(target_len-length(sig),1)];
        end
        
        if ~all(sig == 0)
            sig = sig / sqrt(mean(abs(sig).^2));
            if enable_noise
                sig = awgn(sig, snr, 'measured');
            end
            processed(:,col) = sig;
        end
    end
    valid_signals = processed(:, any(processed,1));
end

%% 特征提取（保持原样）
function [features, valid_idx] = extract_tsne_features(signals)
    valid_idx = find(~all(signals == 0, 1));
    active_signals = signals(:,valid_idx)';
    
    time_features = [real(active_signals), imag(active_signals)];
    freq_signals = fft(active_signals, [], 2);
    freq_features = [abs(freq_signals), angle(freq_signals)];
    features = [time_features, freq_features];
    
    nan_mask = any(isnan(features), 2);
    features(nan_mask,:) = [];
    valid_idx(nan_mask) = [];
end

%% 增强型可视化引擎（新增参数记录）
function visualize_tsne_results(features, labels, output_dir, dpi, perplexity, options)
    % 参数解析
    arguments
        features
        labels
        output_dir
        dpi
        perplexity
        options.SNR_DB = 10
        options.SelectedDevices = 5
        options.ActualDevices = 5
        options.Seed = 2023
    end
    
    % 生成参数化文件名
    file_suffix = sprintf('SNR%d_Sel%d_Act%d_Seed%d',...
        options.SNR_DB, options.SelectedDevices, options.ActualDevices, options.Seed);
    
    % 创建带时间戳的目录
    viz_dir = fullfile(output_dir, 'TSNE_Plots');
    if ~exist(viz_dir, 'dir'), mkdir(viz_dir); end
    
    % 降维（保持随机一致性）
    rng(options.Seed);
    proj_2d = tsne(features, 'NumDimensions', 2, 'Perplexity', perplexity);
    proj_3d = tsne(features, 'NumDimensions', 3, 'Perplexity', perplexity);
    
    % 生成参数化标题
    title_base = @(dim) sprintf('时序轨迹图IQ信号t-SNE %dD投影\nSNR: %ddB | 设备: 选择%d/有效%d | 种子: %d',...
        dim, options.SNR_DB, options.SelectedDevices, options.ActualDevices, options.Seed);

    % 2D可视化
    figure('Position', [100 100 800 600], 'Visible', 'off');
    gscatter(proj_2d(:,1), proj_2d(:,2), grp2idx(labels), lines(length(unique(labels))), '.', 12);
    title(title_base(2));
    exportgraphics(gcf, fullfile(viz_dir, [file_prefix '_2D.png']), 'Resolution', dpi);

    % 3D可视化（新增视角控制）
    figure('Position', [100 100 800 600], 'Visible', 'off');
    scatter3(proj_3d(:,1), proj_3d(:,2), proj_3d(:,3), 12, grp2idx(labels), 'filled');
    title(title_base(3));
    view(-37.5, 30);  % 优化视角参数
    grid on; rotate3d on;
    exportgraphics(gcf, fullfile(viz_dir, [file_prefix '_3D.png']), 'Resolution', dpi);
    close all;
end