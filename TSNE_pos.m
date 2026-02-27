%% 参数配置
input_folder = '..\los_data';        % 输入数据文件夹
output_root = 'TSNE';                % 输出根目录
enable_noise = true;                 % 噪声使能开关
SNR_dB = 0;                         % 信噪比设置
group_size = 320;                    % 每个设备分析的信号数
tsne_perplexity = 30;                % t-SNE困惑度参数
resolution = 300;                    % 输出图像DPI
random_seed = 88;                    % 固定随机种子
num_selected_devices = 20;           % 随机选择的设备数量

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

%% 数据预处理管道（最终正确版本）
[feature_matrix, device_labels] = deal([]);

for d = selected_indices
    % 数据加载
    [~, dev_name] = fileparts(mat_files(d).name);
    try
        data = load(fullfile(input_folder, mat_files(d).name));
        raw_signals = data.data_Ineed;  % 维度: [时间点 × 样本数]
    catch
        fprintf('[%s] 数据加载失败\n', dev_name);
        continue;
    end

    %% 严格随机抽样（确保选择有效信号）
    % 先过滤全零信号
    valid_signals = find(~all(raw_signals == 0, 1));
    if length(valid_signals) < group_size
        fprintf('[%s] 有效信号不足: %d < %d\n', dev_name, length(valid_signals), group_size);
        continue;
    end
    
    rand_samples = valid_signals(randperm(length(valid_signals), group_size));
    
    %% 数据清洗（不再减少信号数）
    selected_signals = raw_signals(:, rand_samples);  % 维度: [时间点 × group_size]
    [processed_signals, ~] = data_cleaning_pipeline(selected_signals, enable_noise, SNR_dB);
    
    %% 特征提取
    [features, valid_mask] = feature_extraction_with_validation(processed_signals);
    
    % 合并数据
    feature_matrix = [feature_matrix; features];
    device_labels = [device_labels; repmat({dev_name}, size(features,1), 1)];
    fprintf('[%s] 有效样本: %d/%d\n', dev_name, size(features,2)/4, group_size);
end

%% 数据完整性检查
assert(~isempty(feature_matrix), '错误：所有选中设备均无有效数据！');
actual_devices = length(unique(device_labels));

%% 降维分析（保持不变）
projection_2d = tsne(feature_matrix, 'NumDimensions', 2, 'Perplexity', tsne_perplexity);
projection_3d = tsne(feature_matrix, 'NumDimensions', 3, 'Perplexity', tsne_perplexity);

%% 可视化输出（完全保留原引擎）
visualization_engine(output_root, resolution, device_labels, projection_2d, projection_3d,...
    SNR_dB, num_selected_devices, actual_devices, random_seed);

%% 数据清洗管道（确保不减少信号数量）
function [processed, valid_idx] = data_cleaning_pipeline(signals, noise_flag, snr)
    % 仅过滤全零列，不删除其他列
    valid_idx = find(~all(signals == 0, 1));
    
    % 如果有效信号不足，填充随机噪声
    if length(valid_idx) < size(signals,2)
        missing = size(signals,2) - length(valid_idx);
        valid_idx = [valid_idx, randperm(size(signals,2), missing)];
    end
    
    active_signals = signals(:, valid_idx);
    
    % 噪声注入
    processed = active_signals;
    if noise_flag
        for col = 1:size(active_signals,2)
            processed(:,col) = awgn(active_signals(:,col), snr, 'measured');
        end
    end
end

%% 特征提取（动态维度适应）
function [features, valid_mask] = feature_extraction_with_validation(signals)
    % 动态计算实际信号数
    actual_group_size = size(signals,2);
    
    % 预分配精确维度
    time_features = zeros(size(signals,1), 2*actual_group_size);
    
    for t = 1:size(signals,1)
        iq_values = signals(t, :);
        time_features(t, :) = [real(iq_values), imag(iq_values)];
    end
    
    % 频域特征（可选）
    try
        freq_signals = fft(signals, [], 2);
        freq_features = [abs(freq_signals), angle(freq_signals)];
        features = [time_features, freq_features];
    catch
        features = time_features;
    end
    
    % 有效性检查
    valid_mask = ~any(isnan(features) | isinf(features), 2);
    features = features(valid_mask, :);
end

%% 增强型可视化引擎（统一风格）
function visualization_engine(output_root, dpi, labels, proj2d, proj3d, snr, num_selected, actual_num, seed)
    % 生成唯一颜色调色板
    [group_idx, group_names] = grp2idx(labels);
    num_groups = length(group_names);
    custom_colors = hsv(max(num_groups, 10));
    custom_colors = custom_colors(1:num_groups, :);
    
    % 文件命名逻辑
    file_prefix = sprintf('SNR%d_Sel%d_Act%d_Seed%d', snr, num_selected, actual_num, seed);
    viz_dir = fullfile(output_root, 'TSNE_Pos');
    if ~exist(viz_dir, 'dir')
        mkdir(viz_dir);
    end

    % 统一标题生成器（完全兼容原格式）
    title_str = @(dim) sprintf('长时序轨迹图-IQ信号t-SNE %dD投影\nSNR: %ddB | 选择/有效设备: %d/%d | 随机种子: %d',...
        dim, snr, num_selected, actual_num, seed);

    %% 2D可视化（颜色升级）
    fig = figure('Position', [100 100 1000 600], 'Visible', 'off');
    gscatter(proj2d(:,1), proj2d(:,2), group_idx, custom_colors, '.', 15);
    legend(group_names, 'Location', 'bestoutside', 'Interpreter', 'none');
    title(title_str(2));
    exportgraphics(fig, fullfile(viz_dir, [file_prefix '_2D.png']), 'Resolution', dpi);
    close(fig);

    %% 3D双模式输出（静态+交互）
    % 静态3D输出
    fig = figure('Position', [100 100 1200 800], 'Visible', 'off');
    plot_enhanced_3d(proj3d, group_idx, group_names, custom_colors);
    title(title_str(3));
    exportgraphics(fig, fullfile(viz_dir, [file_prefix '_3D.png']), 'Resolution', dpi);
    close(fig);
    
    % 交互式3D输出
    fig = figure('Position', [100 100 1200 800], 'Visible', 'on');
    plot_enhanced_3d(proj3d, group_idx, group_names, custom_colors);
    title({title_str(3), '操作提示: 左键旋转 | 滚轮缩放 | 右键平移'}); % 多行标题
    savefig(fig, fullfile(viz_dir, [file_prefix '_3D.fig']));
    
    %% 控制台状态报告
    fprintf('\n[可视化完成] 输出文件:\n');
    fprintf('2D图: %s\n', fullfile(viz_dir, [file_prefix '_2D.png']));
    fprintf('3D静态图: %s\n', fullfile(viz_dir, [file_prefix '_3D.png']));
    fprintf('3D交互图: %s\n', fullfile(viz_dir, [file_prefix '_3D.fig']));
end

%% 增强型3D绘图核心
function plot_enhanced_3d(proj3d, group_idx, group_names, colors)
    hold on;
    h = gobjects(length(group_names), 1);
    for i = 1:length(group_names)
        mask = (group_idx == i);
        h(i) = scatter3(proj3d(mask,1), proj3d(mask,2), proj3d(mask,3),...
            20, 'filled',...
            'MarkerFaceColor', colors(i,:),...
            'MarkerEdgeColor', 'k',...
            'LineWidth', 0.3,...
            'DisplayName', group_names{i});
    end
    legend(h, 'Interpreter', 'none', 'Location', 'bestoutside');
    grid on; 
    rotate3d on;
    view(-37.5, 30);
    hold off;
end