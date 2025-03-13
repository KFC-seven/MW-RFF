%% 参数配置
input_folder = '..\los_nonht';    % 输入.mat文件所在文件夹
output_root = 'IQ_signal_21';      % 输出根目录
group_size = 320;                 % 每组包含的前导码数量
line_style = '-';                 % 连线样式
line_width = 0.5;                 % 线宽
line_color = [0, 0.4470, 0.7410]; % 轨迹线颜色

%% 初始化处理环境
clc; close all;

%% 获取设备文件列表
mat_files = dir(fullfile(input_folder, '*.mat'));
num_devices = length(mat_files);

%% 主处理循环
for d = 1:num_devices
    [~, dev_name] = fileparts(mat_files(d).name);
    fprintf('【开始处理】设备: %s (%d/%d)\n', dev_name, d, num_devices);
    
    %% --- 创建指定目录结构 ---
    device_dir = fullfile(output_root, dev_name);
    output_dir = fullfile(device_dir, 'trajectory_pos'); % 核心修改点
    if ~exist(output_dir, 'dir')
        mkdir(output_dir); % 自动创建不存在的目录层级
    end
    
    %% --- 数据加载与处理 ---
    try
        load(fullfile(input_folder, mat_files(d).name), 'data_Ineed');
        data_transposed = data_Ineed.'; % 转置为 [前导码 × 采样点]
        total_preamble = size(data_transposed, 1);
        
        %% 分组处理
        num_groups = ceil(total_preamble / group_size);
        for g = 1:num_groups
            start_idx = (g-1)*group_size + 1;
            end_idx = min(g*group_size, total_preamble);
            current_group = data_transposed(start_idx:end_idx, :);
            
            %% 处理组内每个前导码
            for local_idx = 1:size(current_group, 1)
                % 生成轨迹图
                fig = figure('Visible', 'off', 'Position', [100, 100, 256, 256]);
                axes('Position', [0 0 1 1], 'Visible', 'off');
                
                % 绘制时序轨迹
                preamble_data = current_group(local_idx, :);
                plot(real(preamble_data), imag(preamble_data),...
                    'LineStyle', line_style,...
                    'LineWidth', line_width,...
                    'Color', line_color);
                xlim([-3 3]); ylim([-3 3]);
                
                % 保存文件
                save_name = fullfile(output_dir,...
                    sprintf('group%02d_%04d.png', g, local_idx));
                exportgraphics(fig, save_name, 'Resolution', 300);
                close(fig);
            end
            fprintf('  完成组 %02d: %d/%d 前导码\n', g, size(current_group,1), group_size);
        end
    catch ME
        warning(ME.identifier, '处理失败: %s', ME.message);
    end
end

fprintf('\n处理完成！文件结构验证:\n%s\n',...
    fullfile(output_root, 'device_A', 'trajectory_pos'));