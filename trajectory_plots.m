%% 参数配置
input_folder = '..\los_data';     % 输入.mat文件所在文件夹
output_root = '..\IQ_signal_21_10dB';  % 输出根目录
lag_samples = 5;                  % 滞后点数
enable_noise = true;             % 是否添加噪声（true/false）
SNR_dB = 10;                      % 噪声信噪比
line_style = '-';                 % 连线样式：'-'实线 | '--'虚线 | ':'点线
line_width = 0.5;                 % 线宽（建议0.5-1.5）
line_color = [0, 0.4470, 0.7410]; % 轨迹线颜色（RGB向量）

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
        output_dir = fullfile(output_root, dev_name, 'trajectory_plots');
        if ~exist(output_dir, 'dir')
            mkdir(output_dir);
        end
        
        % 配置图形参数（重要修改）
        fig = figure('Visible', 'off', 'Position', [100, 100, 256, 256]);
        ax = axes('Parent', fig, 'Position', [0 0 1 1], 'Visible', 'off'); % 全画布坐标系
        
        for sig_idx = 1:total_signals
            %% 信号预处理
            % 处理数据长度不足的情况（新增容错处理）
            max_samples = min(size(data_Ineed,1), 320);
            signal = data_Ineed(1:max_samples, sig_idx);
            if length(signal) < 320
                signal = [signal; zeros(320-length(signal),1)]; % 补零填充
            end
            
            signal = signal / sqrt(mean(abs(signal).^2));
            
            if enable_noise
                signal = awgn(signal, SNR_dB, 'measured');
            end
            
            %% 轨迹计算
            if lag_samples >= length(signal)
                warning('lag_samples(%d) >= signal length(%d)', lag_samples, length(signal));
                lag_samples = 1;
            end
            lagged = signal(lag_samples:end);
            conjugated = signal(1:length(lagged)) .* conj(lagged);
            
             %% ==== 统一画图配置 ====
            fig = figure('Visible', 'off', 'Position', [100, 100, 256, 256], 'Color', 'none');
            ax = axes('Parent', fig, 'Position', [0 0 1 1],...
                'XLim', [-3 3], 'YLim', [-3 3], 'Visible', 'off');
            
            %% ==== 保持原有绘图方式 ====
            plot(ax, real(conjugated), imag(conjugated),...
                'LineStyle', line_style,...
                'LineWidth', line_width,...
                'Color', line_color);
            
            %% ==== 统一保存配置 ====
            save_name = fullfile(output_dir, sprintf('%s_%04d.png', dev_name, sig_idx));
            exportgraphics(fig, save_name, 'Resolution', 300);
            close(fig);
            
            total_processed = total_processed + 1;
        end
    end
    
    %% --- 处理报告 ---
    fprintf('【处理完成】设备: %s\n   生成轨迹图: %d 张\n\n', dev_name, total_processed);
end

fprintf('全部设备处理完成！共生成 %d 个设备的轨迹图\n', num_devices);