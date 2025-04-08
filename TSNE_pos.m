%% 参数配置
input_folder = '..\los_data';        % 输入数据文件夹
output_root = '..\TSNE';             % 输出根目录
enable_noise = true;                 % 噪声使能开关
SNR_dB = 10;                         % 信噪比设置
group_size = 320;                    % 每组样本量
tsne_perplexity = 30;                % t-SNE困惑度参数
resolution = 300;                    % 输出图像DPI
random_seed = 2023;                  % 固定随机种子

%% 初始化环境
clc; close all; 
rng(random_seed, 'twister'); % 固定随机数生成器

%% 获取设备列表
mat_files = dir(fullfile(input_folder, '*.mat'));
num_devices = length(mat_files);
fprintf('发现%d个设备数据\n', num_devices);

%% 数据预处理管道
[feature_matrix, device_labels] = deal([]);

for d = 1:num_devices
    % 数据加载
    [~, dev_name] = fileparts(mat_files(d).name);
    try
        data = load(fullfile(input_folder, mat_files(d).name));
        raw_signals = data.data_Ineed.';
    catch
        fprintf('[%s] 数据加载失败\n', dev_name);
        continue;
    end
    
    %% 严格随机抽样（新增核心逻辑）
    num_total = size(raw_signals, 1);
    
    % 检查样本量是否足够
    if num_total < group_size
        fprintf('[%s] 前导码不足: %d < %d\n', dev_name, num_total, group_size);
        continue;
    end
    
    % 生成可重复的无放回随机索引
    rand_idx = randperm(num_total, group_size); 
    stable_group = raw_signals(rand_idx, :);
    
    %% 数据处理流程
    [processed_signals, clean_idx] = data_cleaning_pipeline(stable_group, enable_noise, SNR_dB);
    [features, valid_features] = feature_extraction_with_validation(processed_signals);
    final_valid_idx = clean_idx(valid_features);
    
    % 数据收集
    if ~isempty(features)
        feature_matrix = [feature_matrix; features];
        device_labels = [device_labels; repmat({dev_name}, size(features,1), 1)];
        fprintf('[%s] 有效样本: %d/%d\n', dev_name, length(final_valid_idx), group_size);
    else
        fprintf('[%s] 无有效数据\n', dev_name);
    end
end

%% 数据完整性检查
assert(size(feature_matrix,1) == length(device_labels),...
    '维度不匹配: 特征矩阵(%d) ≠ 标签数(%d)',...
    size(feature_matrix,1), length(device_labels));

%% 降维分析
fprintf('\n开始t-SNE降维...\n');
projection_2d = tsne(feature_matrix, 'NumDimensions', 2, 'Perplexity', tsne_perplexity);
projection_3d = tsne(feature_matrix, 'NumDimensions', 3, 'Perplexity', tsne_perplexity);

%% 可视化输出
visualization_engine(output_root, resolution, device_labels, projection_2d, projection_3d);

fprintf('\n处理完成! 结果目录: %s\n', output_root);

%% 数据清洗管道（新增）
function [processed, valid_idx] = data_cleaning_pipeline(signals, noise_flag, snr)
    % 初步过滤全零信号
    valid_idx = find(~all(signals == 0, 2));
    active_signals = signals(valid_idx, :);
    
    % 噪声处理（串行模式）
    processed = active_signals;
    if noise_flag
        for i = 1:size(active_signals, 1)
            processed(i,:) = awgn(active_signals(i,:), snr, 'measured');
        end
    end
    
    % 二次过滤异常值
    nan_mask = any(isnan(processed), 2);
    processed(nan_mask,:) = [];
    valid_idx(nan_mask) = [];
end

%% 增强型特征提取（新增有效性验证）
function [features, valid_mask] = feature_extraction_with_validation(signals)
    % 时域特征
    time_features = [real(signals), imag(signals)];
    
    % 频域特征（增加异常捕获）
    try
        freq_signals = fft(signals, [], 2);
        freq_features = [abs(freq_signals), angle(freq_signals)];
    catch
        freq_features = [];
    end
    
    % 特征融合与验证
    features = [time_features, freq_features];
    
    % 过滤包含NaN/Inf的特征
    valid_mask = ~any(isnan(features) | isinf(features), 2);
    features = features(valid_mask,:);
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
    title('长时序轨迹图-IQ信号t-SNE 2D投影', 'FontSize', 12);
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
    exportgraphics(fig, fullfile(viz_dir, '长时序轨迹图-2D_TSNE.png'), 'Resolution', dpi);
    
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
    title('长时序轨迹图-IQ信号t-SNE 3D投影', 'FontSize', 12);
    
    % 添加颜色说明
    colorbar('Ticks', linspace(0,1,num_devices), ...
             'TickLabels', legend_labels, ...
             'Direction', 'reverse', ...
             'FontSize', 8);
    
    % 保存输出
    exportgraphics(fig, fullfile(viz_dir, '长时序轨迹图-3D_TSNE.png'), 'Resolution', dpi);
    close all;
end