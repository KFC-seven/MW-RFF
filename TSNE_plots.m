%% 参数配置（保持原样）
input_folder = '..\los_data';
output_root = '..\TSNE_Visualization';
enable_noise = false;
SNR_dB = 10;
target_length = 320;
tsne_perplexity = 30;
resolution = 300;

%% 初始化环境
clc; close all; rng('default');
mkdir(output_root);

%% 增强型数据管道（新增数据校验）
[feature_matrix, device_labels] = deal([]);

for d = 1:length(dir(fullfile(input_folder, '*.mat')))
    % 数据加载（新增有效性检查）
    [~, dev_name] = fileparts(mat_files(d).name);
    try
        data = load(fullfile(input_folder, mat_files(d).name));
        raw_data = data.data_Ineed;
    catch
        fprintf('[SKIP] 设备%s数据异常\n', dev_name);
        continue; 
    end
    
    % 信号预处理（强化容错）
    valid_signals = process_iq_signals(raw_data, target_length, enable_noise, SNR_dB);
    
    % 特征提取（新增空值过滤）
    [features, valid_idx] = extract_tsne_features(valid_signals);
    
    % 数据收集（强制维度对齐）
    if ~isempty(features)
        feature_matrix = [feature_matrix; features];
        device_labels = [device_labels; repmat({dev_name}, size(features,1), 1)];
    end
end

%% 数据完整性检查（关键新增）
assert(size(feature_matrix,1) == length(device_labels),...
    '数据维度不匹配: 特征矩阵%d行 vs 标签%d个',...
    size(feature_matrix,1), length(device_labels));

%% 可视化（保持原样）
visualize_tsne_results(feature_matrix, device_labels, output_root, resolution, tsne_perplexity);

%% 信号处理函数（新增有效性检查）
function valid_signals = process_iq_signals(raw_data, target_len, enable_noise, snr)
    valid_signals = [];
    if isempty(raw_data) || size(raw_data,2) < 1
        return;
    end
    
    processed = zeros(target_len, min(size(raw_data,2),320));
    for col = 1:size(processed,2)
        sig = raw_data(1:min(end,target_len), col);
        if length(sig) < target_len
            sig = [sig; zeros(target_len-length(sig),1)];
        end
        
        % 信号有效性检查（新增）
        if all(sig == 0)
            continue; % 跳过全零信号
        end
        
        sig = sig / sqrt(mean(abs(sig).^2));
        if enable_noise
            sig = awgn(sig, snr, 'measured');
        end
        processed(:,col) = sig;
    end
    valid_signals = processed(:, any(processed,1)); % 自动过滤全零列
end

%% 增强型特征提取（新增空值过滤）
function [features, valid_idx] = extract_tsne_features(signals)
    valid_idx = find(~all(signals == 0, 1)); % 列过滤
    active_signals = signals(:,valid_idx)';
    
    % 时频特征（保持原样）
    time_features = [real(active_signals), imag(active_signals)];
    freq_signals = fft(active_signals, [], 2);
    freq_features = [abs(freq_signals), angle(freq_signals)];
    features = [time_features, freq_features];
    
    % 新增：过滤无效特征
    nan_mask = any(isnan(features), 2);
    features(nan_mask,:) = [];
    valid_idx(nan_mask) = [];
end

%% 可视化引擎
function visualize_tsne_results(features, labels, output_dir, dpi, perplexity)
    % 创建输出目录
    viz_dir = fullfile(output_dir, 'TSNE_Plots');
    if ~exist(viz_dir, 'dir'), mkdir(viz_dir); end
    
    % t-SNE降维
    fprintf('正在进行t-SNE降维...\n');
    rng(123);  % 固定随机种子保证可重复性
    proj_2d = tsne(features, 'NumDimensions', 2, 'Perplexity', perplexity);
    proj_3d = tsne(features, 'NumDimensions', 3, 'Perplexity', perplexity);
    
    % 颜色映射
    [unique_labels, ~, group_ids] = unique(labels);
    colors = lines(length(unique_labels));
    
    % 2D可视化
    figure('Position', [100 100 800 600], 'Visible', 'off');
    gscatter(proj_2d(:,1), proj_2d(:,2), group_ids, colors, '.', 15);
    title('IQ信号t-SNE 2D投影');
    legend(unique_labels, 'Interpreter', 'none', 'Location', 'best');
    exportgraphics(gcf, fullfile(viz_dir, '2D_TSNE.png'), 'Resolution', dpi);
    
    % 3D可视化
    figure('Position', [100 100 800 600], 'Visible', 'off');
    hold on;
    for i = 1:length(unique_labels)
        idx = group_ids == i;
        scatter3(proj_3d(idx,1), proj_3d(idx,2), proj_3d(idx,3),...
                 36, colors(i,:), 'filled');
    end
    view(135, 30); grid on;
    title('IQ信号t-SNE 3D投影');
    legend(unique_labels, 'Interpreter', 'none', 'Location', 'best');
    exportgraphics(gcf, fullfile(viz_dir, '3D_TSNE.png'), 'Resolution', dpi);
    close all;
end