%% 参数配置
input_folder = fullfile('..', 'los_data');
output_root = fullfile('.', 'TSNE');
enable_noise = true;
SNR_dB = 20;
target_length = 320;
tsne_perplexity = 30;
resolution = 300;
fixed_seed = 88;  
num_selected_devices = 20;  % +++ 新增随机选择参数 +++

%% 初始化环境
clc; close all; 
rng(fixed_seed, 'twister');


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
        
    catch ME
        fprintf('[%s] 数据加载失败: %s\n', dev_name, ME.message);
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
            % 修正为能量归一化
            power = mean(abs(sig).^2);
            sig = sig / sqrt(power);
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
%% 增强型可视化引擎（完整标题+交互功能）
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
    
    % 生成参数化标题函数（完全恢复原格式）
    title_base = @(dim) sprintf('时序轨迹图IQ信号t-SNE %dD投影\nSNR: %ddB | 设备: 选择%d/有效%d | 种子: %d',...
        dim, options.SNR_DB, options.SelectedDevices, options.ActualDevices, options.Seed);
    
    % 颜色生成逻辑（保持不变）
    [group_idx, group_names] = grp2idx(labels);
    num_groups = length(group_names);
    custom_colors = hsv(max(num_groups, 10));
    custom_colors = custom_colors(1:num_groups, :);
    
    % 文件命名逻辑（保持不变）
    file_suffix = sprintf('SNR%d_Sel%d_Act%d_Seed%d',...
        options.SNR_DB, options.SelectedDevices, options.ActualDevices, options.Seed);
    viz_dir = fullfile(output_dir, 'TSNE_Plots');
    if ~exist(viz_dir, 'dir')
        mkdir(viz_dir); 
    end
    
    %% 2D可视化（完全恢复原始标题）
    rng(options.Seed, 'twister');
    proj_2d = tsne(features, 'NumDimensions', 2, 'Perplexity', perplexity);
    
    fig_2d = figure('Position', [100 100 1000 600], 'Visible', 'off');
    gscatter(proj_2d(:,1), proj_2d(:,2), group_idx, custom_colors, '.', 15);
    legend(group_names, 'Location', 'bestoutside', 'Interpreter', 'none');
    title(title_base(2)); % 使用原标题生成器
    exportgraphics(fig_2d, fullfile(viz_dir, [file_suffix '_2D.png']), 'Resolution', dpi);
    close(fig_2d);

    %% 3D双模式输出（保持原标题+新增交互）
    rng(options.Seed, 'twister');
    proj_3d = tsne(features, 'NumDimensions', 3, 'Perplexity', perplexity);
    
    % 静态3D输出（原样恢复）
    fig_3d_static = figure('Position', [100 100 1200 800], 'Visible', 'off');
    plot_3d_scatter(proj_3d, group_idx, group_names, custom_colors);
    title(title_base(3)); % 使用原标题生成器
    exportgraphics(fig_3d_static, fullfile(viz_dir, [file_suffix '_3D.png']), 'Resolution', dpi);
    close(fig_3d_static);
    
    % 交互式3D输出（在原标题基础上添加操作提示）
    fig_3d_interactive = figure('Position', [100 100 1200 800], 'Visible', 'on');
    plot_3d_scatter(proj_3d, group_idx, group_names, custom_colors);
    title({title_base(3), '按住鼠标左键拖动旋转 | 滚轮缩放 | 右键平移'}); % 添加操作提示
    savefig(fig_3d_interactive, fullfile(viz_dir, [file_suffix '_3D.fig']));
    
    %% 控制台输出（显示完整文件路径）
    fprintf('\n=== 可视化输出 ===\n');
    fprintf('2D图: %s\n', fullfile(viz_dir, [file_suffix '_2D.png']));
    fprintf('3D静态图: %s\n', fullfile(viz_dir, [file_suffix '_3D.png']));
    fprintf('3D交互图: %s\n', fullfile(viz_dir, [file_suffix '_3D.fig']));
end

%% 3D绘图函数（保持不变）
function plot_3d_scatter(proj_3d, group_idx, group_names, colors)
    hold on;
    h = gobjects(length(group_names), 1);
    for i = 1:length(group_names)
        mask = (group_idx == i);
        h(i) = scatter3(proj_3d(mask,1), proj_3d(mask,2), proj_3d(mask,3),...
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