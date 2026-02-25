% Simulation of ffd network
% parameters

tau_m=0.02; V_th=20; V_r=0; tau_r=0.002;
sigma=2; I_0=0; V_ref=50;
tau_a=0.005; beta1=-2.74; beta2=0; delta_a=0; A=0;
f_fun=@(v)-v+I_0; F_fun=@(a)a;
g_fun=@(a)-a; G_fun=@(v)A*v;
N_n=2; t_dur=100;dt=1e-4;
c=0.5;

%simulation
t_ref=zeros(floor(t_dur/dt)+1,N_n);
V=zeros(floor(t_dur/dt)+1,N_n);
V(1,:)=V_th*rand(1.,N_n);
L1_output=zeros(floor(t_dur/dt)+1,N_n);
for i=2:floor(t_dur/dt)+1
    if i/10000==floor(i/10000)
        i
    end
    I=I_0;
    %检验不应期
    ind_ref=(t_ref(i,:)>0);
    t_ref(i+1,ind_ref)=t_ref(i,ind_ref)-dt;
    V(i,ind_ref & t_ref(i,:)<=0)=V_r; %在此刻从不应期中恢复
    % Generate noise
    priv_noise=randn(1,N_n);
    public_noise=randn();
    noise=sqrt((1-c))*priv_noise+sqrt(c)*public_noise;
    %演化V1和V2
    dV=dt/tau_m*(f_fun(V(i,:))+sigma*noise/(sqrt(dt)));
    V(i+1,~ind_ref)=V(i,~ind_ref)+dV(~ind_ref);
    % %检查放电
    % ind=find(V(i,:)>=V_th & t_ref(i,:)<=0);
    % L1_output(i,:)=(V(i,:)>=V_th & t_ref(i,:)<=0);
    % t_ref(i+1,ind)=tau_r;
    % V(i+1,ind)=V_r;
end

%% calculate P(v_1,v_2)

% 示例数据：N×2矩阵，第一列是X，第二列是Y
data = V(t_ref(:,1)<=0 & t_ref(:,2)<=0,:);  % 生成10000个标准正态分布点
data=data(100000:end,:);

% 创建二维直方图
figure;
h = histogram2(data(:,1), data(:,2), ...
    'BinMethod', 'auto', ...  % 自动选择分箱
    'DisplayStyle', 'tile', ...  % 瓦片显示
    'ShowEmptyBins', 'on', ...
    'EdgeColor', 'none', ...
    'Normalization', 'pdf');
colorbar;
xlabel('X');
ylabel('Y');
title('二维概率密度直方图');

% 获取概率密度值
counts = h.Values;  % 每个bin的计数
bin_centers_x = h.XBinEdges(1:end-1) + h.BinWidth(1)/2;
bin_centers_y = h.YBinEdges(1:end-1) + h.BinWidth(2)/2;

% 转换为概率密度（积分和为1）
total_count = sum(counts, 'all');
bin_area = h.BinWidth(1) * h.BinWidth(2);
pdf_hist = counts / (total_count * bin_area);
