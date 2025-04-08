%% 参数配置
input_folder = '..\los_data';
output_root = '..\TSNE';
enable_noise = true;
SNR_dB = 10;
target_length = 320;
tsne_perplexity = 30;
resolution = 300;
fixed_seed = 2023;  % 新增固定随机种子参数

%% 初始化环境
clc; close all; 
rng(fixed_seed, 'twister');  % 固定随机种子
mkdir(output_root);

%% 增强型数据管道
[feature_matrix, device_labels] = deal([]);
mat_files = dir(fullfile(input_folder, '*.mat'));

for d = 1:length(mat_files)
    [~, dev_name] = fileparts(mat_files(d).name);
    try
        % 数据加载与校验
        data = load(fullfile(input_folder, mat_files(d).name));
        raw_data = data.data_Ineed;
        
        % 新增随机抽样逻辑
        num_signals = size(raw_data, 2);
        if num_signals < target_length
            fprintf('[%s] 信号不足: %d < %d\n', dev_name, num_signals, target_length);
            continue;
        end
        
        % 固定种子随机抽样
        rand_idx = randperm(num_signals, target_length);
        selected_data = raw_data(:, rand_idx);
        
    catch
        fprintf('[%s] 数据加载失败\n', dev_name);
        continue; 
    end
    
    % 信号预处理（使用抽样数据）
    valid_signals = process_iq_signals(selected_data, target_length, enable_noise, SNR_dB);
    
    % 特征提取
    [features, valid_idx] = extract_tsne_features(valid_signals);
    
    % 数据收集
    if ~isempty(features)
        feature_matrix = [feature_matrix; features];
        device_labels = [device_labels; repmat({dev_name}, size(features,1), 1)];
        fprintf('[%s] 有效样本: %d/%d\n', dev_name, size(features,1), target_length);
    end
end

%% 数据完整性检查
assert(size(feature_matrix,1) == length(device_labels),...
    '维度不匹配: 特征矩阵(%d) ≠ 标签数(%d)',...
    size(feature_matrix,1), length(device_labels));

%% 可视化（保持原样）
visualize_tsne_results(feature_matrix, device_labels, output_root, resolution, tsne_perplexity);

%% 信号处理函数（优化后）
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

%% 统一风格的可视化引擎
function visualization_engine(output_root, dpi, labels, proj2d, proj3d)
    % 创建输出目录dangchu
    viz_dir = fullfile(output_root, 'TSNE_Pos');
    if ~exist(viz_dir, 'dir')
        mkdir(viz_dir);
    end
    
    % 生成唯一颜色映射
    [unique_labels, ~, group_ids] = unique(labels);
    num_devices = length(unique_labels);
    
    % 自定义颜色生成策略
    if num_devices <= 10
        % 小规模设备使用高对比度颜色
        color_palette = lines(num_devices);
    else
        % 大规模设备使用HSV色相环（避免相近颜色）
        hue = linspace(0, 1, num_devices+1)';
        hue = hue(1:end-1);
        color_palette = hsv2rgb([hue, ones(num_devices,1), 0.85*ones(num_devices,1)]);
        
        % 打乱色相顺序避免连续颜色相似
        rand_order = randperm(num_devices);
        color_palette = color_palette(rand_order, :);
    end
    
    %% 2D可视化优化
    fig = figure('Position', [100 100 1200 800], 'Visible', 'off');
    
    % 主图区域
    subplot(2,1,1);
    gscatter(proj2d(:,1), proj2d(:,2), group_ids, color_palette, '.', 20);
    title('时序轨迹图-IQ信号t-SNE 2D投影', 'FontSize', 12);
    grid minor;
    
    % 图例区域
    subplot(2,1,2);
    axis off;
    legend_labels = cellfun(@(x) strrep(x, '_', '\_'), unique_labels, 'UniformOutput', false);
    legend(legend_labels, ...
        'Interpreter', 'none', ...
        'NumColumns', 3, ...
        'FontSize', 9, ...
        'Box', 'off');
    
    % 保存输出
    exportgraphics(fig, fullfile(viz_dir, '时序轨迹图-2D_TSNE.png'), 'Resolution', dpi);
    
    %% 3D可视化优化
    fig = figure('Position', [100 100 1200 800], 'Visible', 'off');
    
    % 三维散点图
    ax = subplot(1,1,1);
    hold on;
    for i = 1:num_devices
        mask = group_ids == i;
        scatter3(proj3d(mask,1), proj3d(mask,2), proj3d(mask,3),...
                 45, color_palette(i,:), 'filled', ...
                 'MarkerEdgeColor', [0.2 0.2 0.2], ...
                 'LineWidth', 0.3);
    end
    view(135, 30);
    grid on;
    title('时序轨迹图-IQ信号t-SNE 3D投影', 'FontSize', 12);
    
    % 添加颜色说明
    colorbar('Ticks', linspace(0,1,num_devices), ...
             'TickLabels', legend_labels, ...
             'Direction', 'reverse', ...
             'FontSize', 8);
    
    % 保存输出
    exportgraphics(fig, fullfile(viz_dir, '时序轨迹图-3D_TSNE.png'), 'Resolution', dpi);
    close all;
end