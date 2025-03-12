num = 0; % 初始化设备编号
% % 创建一个新的 figure
 figure;
% 
% 设置 figure 的大小为 64x64 像素
    set(gcf, 'Position', [100, 100, 256, 256]);  % [left, bottom, width, height]
for i = 0:20
    
    my_rff = zeros(320, 1); % 初始化rff数组
    
    try
        % 加载信号数据
        load("D:\users\long\los_nonht\dev" + i + ".mat"); 
    catch
        num = num + 1;
        continue;
    end
    arr_len(i+1, 1) = size(data_Ineed, 2); % 记录数据维度
    if size(data_Ineed, 2) < 1
        continue;
    end

    % 循环处理信号
    for j = 1:1000
        signal = data_Ineed(1:320, j);
        
        % 计算平均功率
        instant_power = abs(signal).^2;
        average_power = mean(instant_power);

        % 归一化
        normalization_factor = 1 / sqrt(average_power);
        normalized_signal0 = signal * normalization_factor;
            SNR = 20;
        normalized_signal0 = awgn(normalized_signal0,SNR,"measured");

%         % DCTF 过采样
%         oversample_factor = 20; % 过采样因子
%         z_oversampled = interp(normalized_signal0, oversample_factor);
%         normalized_signal0 = z_oversampled;



% % %添加频偏 % 假设你已经收到的信号是320个点的复数信号
% % % fs = 1000; % 采样频率，单位：Hz
% % % N = 320;   % 信号的长度（320个点）
% % % delta_f = 1; % 固定频偏，单位：Hz
% % % 
% % % % 假设你已有的WiFi前导码信号
% % % t = (0:N-1) / fs; % 时间向量，长度为320，采样频率fs
% % % signal = normalized_signal0; % 示例复数信号（请替换为你的实际信号）
% % % 
% % % % 添加固定频偏
% % % signal_with_offset = signal; % 初始化带有频偏的信号，先赋值为原始信号
% % % for n = 1:N
% % %     % 计算每个点的频偏因子
% % %     frequency_offset_factor = exp(1i * 2 * pi * delta_f * t(n)); 
% % %     % 给当前点添加频偏
% % %     signal_with_offset(n) = signal(n) * frequency_offset_factor;
% % % end
% % % normalized_signal0 = signal_with_offset;




% % 过采样因子
% oversample_factor = 1;  % 比如将信号过采样20倍
% 
% % 分离实部和虚部
% real_signal = real(normalized_signal0);
% imag_signal = imag(normalized_signal0);
% 
% % 对实部和虚部分别进行过采样
% real_oversampled = interp(real_signal, oversample_factor);
% imag_oversampled = interp(imag_signal, oversample_factor);
% 
% % 将过采样后的实部和虚部组合成复数信号
% normalized_signal0 = real_oversampled + 1i * imag_oversampled;

        % 滞后信号计算
        x = 5;
        laggedSignal = normalized_signal0(x:end);

        % 共轭相乘
        conjugatedProduct = normalized_signal0(1:length(laggedSignal)) .* conj(laggedSignal);

        % 创建隐藏的图形窗口
        % fig = figure('Visible', 'off'); % 不显示图像窗口
         
        % 使用基础绘图函数绘制复数信号的 IQ 图
       plot(conjugatedProduct, '.', 'MarkerSize', 1);
        xlim([-3 3]);
        ylim([-3 3]);
%         xlabel('In-Phase'); % 设置X轴标签
%         ylabel('Quadrature'); % 设置Y轴标签
%         title('IQ Scatter Plot'); % 设置标题






% % % % % % 过采样因子
% % % % % oversample_factor = 20;  % 比如将信号过采样20倍
% % % % % 
% % % % % % 分离实部和虚部
% % % % % real_signal = real(normalized_signal0);
% % % % % imag_signal = imag(normalized_signal0);
% % % % % 
% % % % % % 对实部和虚部分别进行过采样
% % % % % real_oversampled = interp(real_signal, oversample_factor);
% % % % % imag_oversampled = interp(imag_signal, oversample_factor);
% % % % % 
% % % % % % 将过采样后的实部和虚部组合成复数信号
% % % % % normalized_signal0 = real_oversampled + 1i * imag_oversampled;
% % % % % 
% % % % % % 滞后信号计算
% % % % % x = 40;
% % % % % laggedSignal = normalized_signal0(x:end);
% % % % % 
% % % % % % 共轭相乘
% % % % % conjugatedProduct = normalized_signal0(1:length(laggedSignal)) .* conj(laggedSignal);
% % % % % 
% % % % % % 使用hist3计算密度
% % % % % nbins = 300; % 设置直方图的网格数量
% % % % % [nn, ctrs] = hist3([real(conjugatedProduct), imag(conjugatedProduct)], 'Nbins', [nbins nbins]);
% % % % % 
% % % % % % 提取每个维度的中心坐标
% % % % % xcenters = ctrs{1};  % 实部的中心坐标
% % % % % ycenters = ctrs{2};  % 虚部的中心坐标
% % % % % 
% % % % % 
% % % % % 
% % % % % % 使用这些范围来筛选nn数据
% % % % % nn_filtered = nn(x_range, y_range);
% % % % % xcenters_filtered = xcenters(x_range);
% % % % % ycenters_filtered = ycenters(y_range);
% % % % % 
% % % % % % 绘制密度图
% % % % % figure;
% % % % % imagesc(xcenters_filtered, ycenters_filtered, nn_filtered');  % 使用颜色映射显示密度
% % % % % axis xy; % 让y轴从下到上显示
% % % % % colorbar;  % 显示颜色条
% % % % % xlabel('Real Part');
% % % % % ylabel('Imaginary Part');
% % % % % title('Density Plot of Conjugated Product');
% % % % % 
% % % % % % 设置黄蓝色渐变的颜色映射（使用 parula 或 jet）
% % % % % colormap('parula');  % 使用 parula 颜色映射
% % % % % 
% % % % % % 限制x和y轴范围为[-3, 3]，并确保没有白边
% % % % % axis([-3 3 -3 3]);  % 设置x轴和y轴范围，并确保没有白边







% % % % DCTF 过采样
% % % oversample_factor = 20; % 过采样因子
% % % z_oversampled = interp(normalized_signal0, oversample_factor);
% % % normalized_signal0 = z_oversampled;
% % % 
% % % % 滞后信号计算
% % % x = 40;
% % % laggedSignal = normalized_signal0(x:end);
% % % 
% % % % 共轭相乘
% % % conjugatedProduct = normalized_signal0(1:length(laggedSignal)) .* conj(laggedSignal);
% % % 
% % % % 提取实部和虚部
% % % x_vals = real(conjugatedProduct);
% % % y_vals = imag(conjugatedProduct);
% % % 
% % % % 计算密度
% % % nbins = 500; % 网格分辨率
% % % [counts, xedges, yedges] = histcounts2(x_vals, y_vals, nbins);
% % % % 设置背景颜色为蓝色
% % % set(gca, 'Color', 'blue'); % 设置坐标轴背景为蓝色
% % % % 将counts映射为颜色，点数密集的地方颜色更亮
% % % imagesc(xedges, yedges, log(counts')); % 使用对数尺度显示密度
% % % 
% % % % 设置图像属性
% % % axis xy;
% % % colormap('hot'); % 选择热力图颜色
% % % colorbar; % 显示颜色条
% % % 
% % % % 添加标题和标签
% % % xlabel('In-Phase');
% % % ylabel('Quadrature');
% % % title('Density Plot of IQ Scatter');






 x = normalized_signal0;

       
% % % % % % % % %%% 12组多DCTF
% % % % % % % % x = normalized_signal0(1:160);
% % % % % % % % % oversample_factor = 10; % 过采样因子
% % % % % % % % % x = interp(x, oversample_factor);
% % % % % % % % fs = 20e6;  % 采样率 20 MHz
% % % % % % % % N = length(x);  % 信号长度 160
% % % % % % % % % t = (0:N-1) / fs;  % 生成时间索引
% % % % % % % % delta_f = fs / N;  % 频率分辨率 125 kHz
% % % % % % % % 
% % % % % % % % % figure;
% % % % % % % % t = tiledlayout(3, 4, 'TileSpacing', 'compact', 'Padding', 'compact'); % 创建3x4布局
% % % % % % % % 
% % % % % % % % % 自定义每个子图的 x 轴和 y 轴范围（用 cell 数组存储）
% % % % % % % % % x_limits = {[-0.1, 0.1], [-0.3, 0.1], [-0.1, 0.1], [-0.05, 0.15], ...
% % % % % % % % %             [-0.05, 0.05], [-0.05, 0.05], [-0.05, 0.05], [-0.05, 0.05], ...
% % % % % % % % %             [-0.05, 0.25], [-0.15, 0.1], [-0.25, 0.05], [-0.2, 0.1]}; 
% % % % % % % % % 
% % % % % % % % % y_limits = {[-0.4, 0.15], [-0.2, 0.1], [-0.05, 0.15], [-0.1, 0.05], ...
% % % % % % % % %             [-0.1, 0.1], [-0.05, 0.15], [-0.1, 0.05], [-0.1, 0.25], ...
% % % % % % % % %             [-0.15, 0.15], [-0.3, 0.05], [-0.15, 0.2], [-0.1, 0.25]}; 
% % % % % % % % 
% % % % % % % % x_limits = {[-0.15, 0.15], [-0.3, 0.1], [-0.1, 0.1], [-0.1, 0.3], ...
% % % % % % % %             [-0.1, 0.1], [-0.2, 0.1], [-0.15, 0.15], [-0.1, 0.1], ...
% % % % % % % %             [-0.1, 0.25], [-0.15, 0.15], [-0.4, 0.1], [-0.15, 0.15]}; 
% % % % % % % % 
% % % % % % % % y_limits = {[-0.4, 0.15], [-0.15, 0.15], [-0.1, 0.3], [-0.1, 0.1], ...
% % % % % % % %             [-0.15, 0.1], [-0.1, 0.1], [-0.1, 0.1], [-0.1, 0.2], ...
% % % % % % % %             [-0.1, 0.1], [-0.4, 0.15], [-0.2, 0.2], [-0.1, 0.35]};
% % % % % % % % 
% % % % % % % % index = 1; % 索引变量
% % % % % % % % 
% % % % % % % % for kkk = [1:6, 10:15]
% % % % % % % %     f_shift = kkk * 10 * delta_f;
% % % % % % % %     x = normalized_signal0(1:160);
% % % % % % % %     
% % % % % % % %     % 进行频谱搬移
% % % % % % % %     time = (0:N-1) / fs;  % 生成时间索引
% % % % % % % %     shiftx = x .* (exp(1j * 2 * pi * (f_shift) * time))';
% % % % % % % % 
% % % % % % % %     % 设计巴特沃斯低通滤波器
% % % % % % % %     order = 4;
% % % % % % % %     cutoff_freq = 0.7e6;
% % % % % % % %     [b, a] = butter(order, (cutoff_freq / (fs/2)), 'low');
% % % % % % % % 
% % % % % % % %     % 应用滤波器
% % % % % % % %     filtered_shiftx = filter(b, a, shiftx);
% % % % % % % % 
% % % % % % % %     % 计算 FFT
% % % % % % % %     X = fft(x);
% % % % % % % %     Shifted_X = fft(shiftx);
% % % % % % % %     Filtered_X = fft(filtered_shiftx);
% % % % % % % %     time = (0:N-1) / fs;  % 生成时间索引
% % % % % % % % 
% % % % % % % %     % 将滤波后的频谱搬回去
% % % % % % % %     filtered_shiftx = filtered_shiftx .* (exp(1j * 2 * pi * (-f_shift) * time))';
% % % % % % % % 
% % % % % % % %     % 计算共轭相乘
% % % % % % % %     x_lag = 5;
% % % % % % % %     laggedSignal = filtered_shiftx(x_lag:end);
% % % % % % % %     conjugatedProduct = filtered_shiftx(1:length(laggedSignal)) .* conj(laggedSignal);
% % % % % % % % 
% % % % % % % %     % 添加到 tiledlayout
% % % % % % % %     nexttile;
% % % % % % % %     % plot(real(conjugatedProduct), imag(conjugatedProduct), '.', 'MarkerSize', 0.001);
% % % % % % % %     scatter(real(conjugatedProduct), imag(conjugatedProduct), 0.5, 'filled');
% % % % % % % % 
% % % % % % % %     % 隐藏 x 轴和 y 轴的刻度值
% % % % % % % %     xticks([]);  
% % % % % % % %     yticks([]);  
% % % % % % % %     
% % % % % % % %     % 彻底隐藏坐标轴（如果不想显示坐标轴）
% % % % % % % %        axis off;
% % % % % % % % 
% % % % % % % %     % 为每个子图分别设置 x 和 y 轴范围
% % % % % % % %     xlim(x_limits{index}); 
% % % % % % % %     ylim(y_limits{index}); 
% % % % % % % % %  xlim([-0.3 0.3]); 
% % % % % % % % %   ylim([-0.3 0.3]); 
% % % % % % % % 
% % % % % % % %     index = index + 1; % 更新索引
% % % % % % % % end






% % % % STFT        % 设置 STFT 参数
% % % %         window_len = 64;      % 窗口长度 64
% % % %         noverlap = 32;        % 重叠长度 32（50% 重叠）
% % % %         nfft = 512;           % FFT 点数 128
% % % %         fs = 20e6;
% % % %         % 使用 stft 函数进行 STFT 变换
% % % %         [stft_matrix, f, t] = stft(x, fs, 'Window', hamming(window_len), 'OverlapLength', noverlap, 'FFTLength', nfft);
% % % %         
% % % %         % 绘制 STFT 的幅度谱
% % % %         
% % % %         imagesc(t, f, abs(stft_matrix));  % 绘制 STFT 的幅度谱（幅度）



% % % % % 功率谱密度
% % % % fs = 20e6; % 采样率 20 MHz
% % % % N_w = 128; % 窗口大小
% % % % overlap = N_w / 2; % 50% 重叠
% % % % N_fft = 256; % FFT 点数
% % % % 
% % % % [pxx, f] = pwelch(x, hamming(N_w), overlap, N_fft, fs, 'centered');
% % % % 
% % % % %  features = [max(pxx), mean(pxx), var(pxx)]; % 提取最大值、均值、方差
% % % % features = [max(10*log10(pxx(22:234))), mean(10*log10(pxx(22:234))), var(10*log10(pxx(22:234)))]; % 提取最大值、均值、方差
% % % %  plot(features);
% % % % %  plot( 10*log10(pxx));
% % % % %  plot(pxx);
% % % % xlabel('Frequency (MHz)');
% % % % ylabel('Power Spectral Density (dB/Hz)');
% % % % title('WiFi 信号前导码的功率谱密度');
% % % % grid on;
% % % % 
% % % % hold on






% % % % % % %双谱
% % % % % % 
% % % % % %  signal =x; 
% % % % % % 
% % % % % % M = 64;   % 设定窗口长度
% % % % % % L = M/2;  % 设定重叠部分
% % % % % % N = length(signal); % 信号长度
% % % % % % nfft = 512;  % FFT 计算点数
% % % % % % wind = hamming(M);  % Hamming 窗口
% % % % % % nsamp = M;   % 每个窗口的样本数
% % % % % % overlap = L; % 设定重叠部分
% % % % % % 
% % % % % % % 调用 bispecd
% % % % % % [Bspec, waxis] = BISPECD(signal, nfft, wind, nsamp, overlap);

% % 4. 绘制等高线图
% % 归一化频率轴
% w1 = linspace(-0.5, 0.5, size(Bspec,1));
% w2 = linspace(-0.5, 0.5, size(Bspec,2));
% 
% % 绘制双谱等高线图
% figure;
% contour(w1, w2, abs(Bspec), 20);
% xlabel('w1');
% ylabel('w2');
% title('WiFi信号双谱等高线图');
% grid on;
% colormap('parula');
% colorbar;


       % 定义保存路径和文件名
        save_path = "D:\users\long\DCTF_320point_1overSample-20db\train\" + num + "\" + j + ".png";
        
        % 保存图形
        saveas(gcf, save_path);
        % close(fig); % 关闭隐藏的图形窗口
    end
    num = num + 1;
end
