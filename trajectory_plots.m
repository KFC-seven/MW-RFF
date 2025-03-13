%% 参数配置
input_folder = '..\los_nonht';    % 输入.mat文件所在文件夹
output_root = '..\IQ_signal_21';     % 输出根目录
signal_bandwidth = 20e6;          % 20 MHz
enable_noise = false;             % 是否添加噪声（true/false）
SNR_dB = 20;                      % 噪声信噪比
line_style = '-';                 % 连线样式：'-'实线 | '--'虚线 | ':'点线
line_width = 0.5;                 % 线宽（建议0.5-1.5）
line_color = [0, 0.4470, 0.7410]; % 轨迹线颜色（RGB向量）

% 根据信号带宽自动计算最佳滞后点数
tau = round(1/(signal_bandwidth * 1e-6)); % 示例：τ=5 (对应20MHz带宽)
lag_samples = tau;

%% 初始化处理环境
clc; close all;
mkdir(output_root);

%% 获取设备文件列表
mat_files = dir(fullfile(input_folder, '*.mat'));
num_devices = length(mat_files);

%% 主处理循环
for d = 1:num_devices
    total_processed = 0;
    [~, dev_name] = fileparts(mat_files(d).name);
    fprintf('【开始处理】设备: %s (%d/%d)\n', dev_name, d, num_devices);
    
    %% --- 数据加载 ---
    try
        load(fullfile(input_folder, mat_files(d).name), 'data_Ineed');
        if exist('data_Ineed', 'var') && size(data_Ineed, 2) >= 1
            total_signals = size(data_Ineed, 2);
        else
            error('InvalidData:DataStructure', '数据字段缺失');
        end
    catch ME
        warning(ME.identifier, '数据加载失败: %s', ME.message);
        total_signals = 0;
    end

    %% --- 轨迹生成 ---
    if total_signals > 0
        output_dir = fullfile(output_root, dev_name, 'trajectory_plots'); % 修改目录名
        if ~exist(output_dir, 'dir')
            mkdir(output_dir);
        end
        
        fig = figure('Visible', 'off', 'Position', [100, 100, 256, 256]);
        axes('Position', [0 0 1 1], 'Visible', 'off');
        
        for sig_idx = 1:total_signals
            %% 信号预处理
            signal = data_Ineed(1:320, sig_idx);
            signal = signal / sqrt(mean(abs(signal).^2));
            
            if enable_noise
                signal = awgn(signal, SNR_dB, 'measured');
            end
            
            %% 轨迹计算
            lagged = signal(lag_samples:end);
            conjugated = signal(1:length(lagged)) .* conj(lagged);
            
            %% 绘制连线图（核心修改部分）
            clf(fig);
            hold on;
            % 绘制顺序连线
            plot(real(conjugated), imag(conjugated),...
                'LineStyle', line_style,...
                'LineWidth', line_width,...
                'Color', line_color);
            hold off;
            
            % 保持坐标范围一致
            xlim([-3 3]);
            ylim([-3 3]);
            
            %% 保存轨迹图
            save_name = fullfile(output_dir, sprintf('%s_%04d.png', dev_name, sig_idx));
            exportgraphics(fig, save_name, 'Resolution', 300);
            total_processed = total_processed + 1;
        end
        close(fig);
    end
    
    %% --- 处理报告 ---
    fprintf('【处理完成】设备: %s\n   生成轨迹图: %d 张\n\n', dev_name, total_processed);
end

fprintf('全部设备处理完成！共生成 %d 个设备的轨迹图\n', num_devices);