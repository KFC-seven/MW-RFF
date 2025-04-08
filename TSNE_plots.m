%% 参数配置
input_folder = '..\los_data';        % 输入数据文件夹
output_root = '..\TSNE_Visualization'; % 输出根目录
enable_noise = true;                % 噪声使能开关
SNR_dB = 10;                         % 信噪比设置
target_length = 320;                 % 目标信号长度
tsne_perplexity = 30;                % t-SNE困惑度参数
resolution = 300;                    % 输出图像DPI

%% 初始化环境
clc; close all; rng('default');
mkdir(output_root);

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
        raw_data = data.data_Ineed;
    catch
        fprintf('[错误] 设备%s数据加载失败\n', dev_name);
        continue;
    end
    
    % 信号标准化处理
    processed_signals = zeros(target_length, 320);
    for col = 1:min(size(raw_data,2),320)  % 仅处理前320列
        signal = raw_data(:, col);
        
        % 信号截断/补零
        if length(signal) > target_length
            signal = signal(1:target_length);
        elseif length(signal) < target_length
            signal = [signal; zeros(target_length - length(signal),1)];
        end
        
        % 信号归一化
        signal = signal / sqrt(mean(abs(signal).^2));
        
        % 添加噪声
        if enable_noise
            signal = awgn(signal, SNR_dB, 'measured');
        end
        
        processed_signals(:, col) = signal;
    end
    
    % 特征提取
    features = extract_tsne_features(processed_signals.');
    
    % 数据收集
    feature_matrix = [feature_matrix; features];
    device_labels = [device_labels; repmat({dev_name}, size(features,1), 1)];
    
    fprintf('[%s] 已处理%d个样本\n', dev_name, size(features,1));
end

%% t-SNE降维可视化
visualize_tsne_results(feature_matrix, device_labels, output_root, resolution, tsne_perplexity);

fprintf('\n处理完成! 结果目录: %s\n', output_root);

%% 特征提取函数
function features = extract_tsne_features(signals)
    % 时域特征
    time_features = [real(signals), imag(signals)];
    
    % 频域特征
    freq_signals = fft(signals, [], 2);
    freq_features = [abs(freq_signals), angle(freq_signals)];
    
    % 特征融合
    features = [time_features, freq_features];
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