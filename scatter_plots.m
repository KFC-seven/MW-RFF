%% 参数配置
input_folder = '..\los_data';     % 输入.mat文件所在文件夹（相对路径）
output_root = '..\IQ_signal_21_10dB';  % 输出根目录
enable_noise = true;             % 是否添加噪声（true/false）
SNR_dB = 10;                      % 噪声信噪比（启用时有效）
lag_samples = 5;                  % 滞后点数

%% 初始化处理环境
clc; close all;
mkdir(output_root); % 确保输出根目录存在

%% 获取所有设备文件列表
mat_files = dir(fullfile(input_folder, '*.mat'));
num_devices = length(mat_files);

%% 主循环处理每个设备
for d = 1:num_devices
    total_processed = 0; % 当前设备处理计数器
    [~, dev_name] = fileparts(mat_files(d).name);
    fprintf('【开始处理】设备: %s (%d/%d)\n', dev_name, d, num_devices);
    
    %% --- 数据加载与验证 ---
    try
        % 加载数据并验证结构
        load(fullfile(input_folder, mat_files(d).name), 'data_Ineed');
        if exist('data_Ineed', 'var') && size(data_Ineed, 2) >= 1
            total_signals = size(data_Ineed, 2);
        else
            error('InvalidData:DataStructure', '数据字段缺失或无效');
        end
    catch ME
        warning(ME.identifier, '数据加载失败: %s', ME.message); % 标准错误输出
        total_signals = 0;
    end

    %% --- 数据处理（当数据有效时）---
    if total_signals > 0
        % 创建分层输出目录
        output_dir = fullfile(output_root, dev_name, 'scatter_plots');
        if ~exist(output_dir, 'dir')
            mkdir(output_dir);
        end
        
        % 预配置图形参数（提升性能）
        fig = figure('Visible', 'off', 'Position', [100, 100, 256, 256]);
        axes('Position', [0 0 1 1], 'Visible', 'off');
        
        %% 信号处理流水线
        for sig_idx = 1:total_signals
            % 1. 信号截取与归一化（保持不变）
            max_samples = min(size(data_Ineed, 1), 320);
            signal = data_Ineed(1:max_samples, sig_idx);
            if size(signal,1) < 320
                signal = [signal; zeros(320 - size(signal,1), 1)];
            end
            signal = signal / sqrt(mean(abs(signal).^2));
            
            % 2. 噪声注入（保持不变）
            if enable_noise
                signal = awgn(signal, SNR_dB, 'measured');
            end
            
            % 3. 滞后共轭乘积（保持不变）
            if lag_samples >= length(signal)
                warning('lag_samples(%d) >= signal length(%d)', lag_samples, length(signal));
                lag_samples = 1;
            end
            lagged = signal(lag_samples:end);
            conjugated = signal(1:length(lagged)) .* conj(lagged);
            
            % 4. 画图
            fig = figure('Visible', 'off', 'Position', [100, 100, 256, 256], 'Color', 'none');
            ax = axes('Parent', fig, 'Position', [0 0 1 1],...
                'XLim', [-3 3], 'YLim', [-3 3], 'Visible', 'off');
            
            % ==== 保持原有绘图方式 ====
            plot(ax, real(conjugated), imag(conjugated), '.', 'MarkerSize', 2);
            
            % ==== 统一保存配置 ====
            save_name = fullfile(output_dir, sprintf('%s_%04d.png', dev_name, sig_idx));
            exportgraphics(fig, save_name, 'Resolution', 300);
            close(fig);
            
            total_processed = total_processed + 1;
        end
    end
    
    %% --- 处理结果报告 ---
    fprintf('【处理完成】设备: %s\n   生成图片: %d 张\n\n', dev_name, total_processed);
end

%% 最终状态输出
fprintf('全部设备处理完成！共处理 %d 个设备\n', num_devices);