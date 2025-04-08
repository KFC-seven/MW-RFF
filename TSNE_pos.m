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

%% 数据完整性校验（保持原样）
assert(size(feature_matrix, 1) == length(device_labels),...);

%% 降维分析
fprintf('\n开始t-SNE降维...\n');
projection_2d = tsne(feature_matrix, 'NumDimensions', 2, 'Perplexity', tsne_perplexity);
projection_3d = tsne(feature_matrix, 'NumDimensions', 3, 'Perplexity', tsne_perplexity);

%% 可视化输出
visualization_engine(output_root, resolution, device_labels, projection_2d, projection_3d);

fprintf('\n处理完成! 结果目录: %s\n', output_root);

%% 噪声处理模块 (保持原样)
function y = if_enable_noise(x, enable, snr)
    y = x;
    if enable
        parfor i = 1:size(x,1)
            y(i,:) = awgn(x(i,:), snr, 'measured');
        end
    end
end

%% 特征提取模块 (保持原样)
function [features, valid_idx] = extract_features(signals)
    valid_idx = find(~all(signals == 0, 2));  % 过滤全零信号
    active_signals = signals(valid_idx, :);
    
    % 时频特征组合
    time_features = [real(active_signals), imag(active_signals)];
    
    % 频域特征
    freq_signals = fft(active_signals, [], 2);
    freq_features = [abs(freq_signals), angle(freq_signals)];
    
    % 特征融合
    features = [time_features, freq_features];
end

%% 统一风格的可视化引擎
function visualization_engine(output_root, dpi, labels, proj2d, proj3d)
    % 创建输出目录
    viz_dir = fullfile(output_root, 'TSNE_Pos');
    if ~exist(viz_dir, 'dir')
        mkdir(viz_dir);
    end
    
    % 生成颜色映射
    [unique_labels, ~, group_ids] = unique(labels);
    color_palette = lines(length(unique_labels)); % 改用lines配色
    
    %% 2D可视化
    fig = figure('Position', [100 100 800 600], 'Visible', 'off');
    gscatter(proj2d(:,1), proj2d(:,2), group_ids, color_palette, '.', 8); % 统一点标记样式
    title('长时序轨迹图-IQ信号t-SNE 2D投影'); % 中文标题
    legend(unique_labels, 'Interpreter', 'none', 'Location', 'best');
    exportgraphics(fig, fullfile(viz_dir, '2D_TSNE.png'), 'Resolution', dpi);
    
    %% 3D可视化
    fig = figure('Position', [100 100 800 600], 'Visible', 'off');
    hold on;
    for i = 1:length(unique_labels)
        mask = group_ids == i;
        scatter3(proj3d(mask,1), proj3d(mask,2), proj3d(mask,3),...
                 10, color_palette(i,:), 'filled'); % 保持点大小一致
    end
    view(135, 30); % 固定视角参数
    grid on;
    title('长时序轨迹图-IQ信号t-SNE 3D投影'); % 中文标题
    legend(unique_labels, 'Interpreter', 'none', 'Location', 'best');
    exportgraphics(fig, fullfile(viz_dir, '3D_TSNE.png'), 'Resolution', dpi);
    close all;
end