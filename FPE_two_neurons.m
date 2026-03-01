clear; clc; tic;

%% Parameters
fprintf('setting parameters...\n');

% tau_m=0.02; vth=20; vr=0; tref=0.002;
% beta=4; mu=15; vref=50;
% tau_a=0.005; beta1=-2.74; beta2=0; delta_a=0; A=0;
% f_fun=@(v)-v+mu; F_fun=@(a)a;
% g_fun=@(a)-a; G_fun=@(v)A*v;

tau_m=0.02; vth=20; vr=0; tref=0.002;
sigma=2; I=8;
A=0;B=0;
f_fun=@(v1)-v1+I; F_fun=@(v2)B*v2;
g_fun=@(v2)-v2+I; G_fun=@(v1)A*v1;
c=0.5;

p.tau_m=0.02; p.vth=20; p.vr=0; p.tref=0.002;
p.sigma=4; p.I=15;
p.A=0;p.B=0;
p.f_fun=@(v1)-v1+I; p.F_fun=@(v2)B*v2;
p.g_fun=@(v2)-v2+I; p.G_fun=@(v1)A*v1;
p.c=0.5;

N=400;p.N=400;
v2_loc=0:N:(N-1)*N;
v1_loc=1:N;
v0=-40; dv=(vth-v0)/N; p.dv=dv; v1=linspace(v0,vth-dv,N)'; v2=linspace(v0,vth-dv,N)'; 
dt=0.00001;nref=floor(tref/dt);
%% 1D operators (fixed spdiags syntax)


v1v1=sigma^2/tau_m^2/2/dv^2;
v2v2=sigma^2/tau_m^2/2/dv^2;
diag=-2*v1v1-2*v2v2;
c1=-F_fun(v2)/2./tau_m/dv;
c2=-f_fun(v1)/2./tau_m/dv;
c3=c*sigma^2/tau_m/tau_m/4./dv^2;
c4=-g_fun(v2)/tau_m/2./dv;
c5=-G_fun(v1)/2./tau_m/dv;

% L operator
fprintf('Construting L operator...\n');

tic;
row=[];col=[];value=[];
for j=1:N-1
row1=v1_loc(j)+v2_loc; col1=v1_loc(j+1)+v2_loc; value1=c1+c2(j+1)+v1v1;
row2=v1_loc+v2_loc(j); col2=v1_loc+v2_loc(j+1); value2=v2v2+c4(j+1)+c5;
row3=v1_loc(2:end)+v2_loc(j); col3=v1_loc(2:end)-1+v2_loc(j+1); value3=zeros(N-1,1)-c3;
row4=v1_loc(1:end-1)+v2_loc(j); col4=v1_loc(1:end-1)+1+v2_loc(j+1); value4=zeros(N-1,1)+c3;
row=[row row1 row2 row3 row4];
col=[col col1 col2 col3 col4];
value=[value value1' value2' value3' value4'];
end

for j=2:N
    row1=v1_loc(j)+v2_loc; col1=v1_loc(j-1)+v2_loc; value1=-c1-c2(j-1)+v1v1;
    row2=v1_loc+v2_loc(j); col2=v1_loc+v2_loc(j-1); value2=v2v2-c4(j-1)-c5;
    row3=v1_loc(2:end)+v2_loc(j); col3=v1_loc(2:end)-1+v2_loc(j-1); value3=zeros(N-1,1)+c3;
    row4=v1_loc(1:end-1)+v2_loc(j); col4=v1_loc(1:end-1)+1+v2_loc(j-1); value4=zeros(N-1,1)-c3;
    row=[row row1 row2 row3 row4];
    col=[col col1 col2 col3 col4];
    value=[value value1' value2' value3' value4'];
end
L=sparse(row,col,value,N^2,N^2);
L=L+diag*speye(N^2);

%%
fprintf('Construting R operator(F)...\n');
% Reset operator R
nr=floor(vr-v1(1)/dv)+1;
part_r=(vr-v1(1))/dv-nr+1;
%Absorb the firing neurons F
row=1:N; col=N+v2_loc; value=zeros(length(row1),1)+v1v1;
F_1=sparse(row,col,value,N,N^2);

row=1:N; col=v1_loc+v2_loc(N); value=zeros(length(row1),1)+v2v2;
F_2=sparse(row,col,value,N,N^2);
fprintf('Construting R operator(E)...\n');
%Diffusion of a neuron during refractory period of another neuron(E) 
E_1=zeros(N,N,nref+1);
for n=1:nref+1
    E_1(:,:,n)=gen_x_ref(v2,(n-1)*dt,p);
    if n ~= nref+1 %normalization
        E_1(:,:,n)=E_1(:,:,n)./sum(E_1(:,:,n));
    end
end

E_2=zeros(N,N,nref+1);
for n=1:nref+1
    E_2(:,:,n)=gen_x_ref(v1,(n-1)*dt,p);
    if n ~= nref+1 %normalization
        E_2(:,:,n)=E_2(:,:,n)./sum(E_2(:,:,n));
    end
end
fprintf('Construting R operator(F12,F21)...\n');
%Absorb the firing from a Nx1 refractory matrix(F12,F21)
F12=sparse(N,N);F12(nr,N)=v2v2*(1-part_r);F12(nr+1,N)=v2v2*part_r;
F21=sparse(N,N);F21(nr,N)=v1v1*(1-part_r);F21(nr+1,N)=v1v1*part_r;

fprintf('Construting R operator(S)...\n');
%reinsert the probability to subthreshold regime(S)
row1=v1_loc(nr)+v2_loc;col1=1:N;value1=zeros(1,N)+1-part_r;
row2=v1_loc(nr+1)+v2_loc;col2=1:N;value2=zeros(1,N)+part_r;
row=[row1,row2];col=[col1,col2];value=[value1,value2];
S1=sparse(row,col,value,N^2,N);

row1=v1_loc+v2_loc(nr);col1=1:N;value1=zeros(1,N)+1-part_r;
row2=v1_loc+v2_loc(nr+1);col2=1:N;value2=zeros(1,N)+part_r;
row=[row1,row2];col=[col1,col2];value=[value1,value2];
S2=sparse(row,col,value,N^2,N);

fprintf('Construting R operator(summation)...\n');
%calculate components of R
R_1_taur=S1*sparse(E_1(:,:,end))*F_1;
R_2_taur=S2*sparse(E_2(:,:,end))*F_2;

R_12=cell(1,nref);R_21=cell(1,nref);
for i=1:nref
    R_12{i}=S2*sparse(E_2(:,:,i))*F12*sparse(E_1(:,:,i))*F_1;
    R_21{i}=S1*sparse(E_1(:,:,i))*F21*sparse(E_2(:,:,i))*F_2;
end

% calculate R
R1=sparse(N^2,N^2);R2=sparse(N^2,N^2);
for i=1:nref
    R1=R1+R_12{i}*dt;
    R2=R2+R_21{i}*dt;
end
R1=R1+R_1_taur;R2=R2+R_2_taur;

R=R_1_taur+R_2_taur;
% clear x_reff
% clear x_ref

%% Solve stationary state
fprintf('Solving stationary state...\n');
onecol = floor(N^2/2)+nr+1;
One = sparse((1:N^2)', repmat(onecol,N^2,1), ones(N^2,1), N^2,N^2);
% P = (L+R+One)\ones(N^2,1);
P = (L+R_1_taur+R_2_taur+One)\ones(N^2,1);
P = P/sum(P);

P=P/dv/dv;
Ps = R_1_taur*P; Ps(Ps<0)=0;
rate_tmp=sum(Ps)*dv*dv;
Ps=Ps/sum(Ps)/(dv*dv);%Ps=R*P0/rate after normalization.
rate=1/(tref+1/rate_tmp);
P0=P*(1-tref*rate)^2;
fprintf('rate=%.3f Hz\n', rate);

%% Power spectrum
n_omega=21; omega=[0,exp(linspace(log(pi),log(1000*2*pi),n_omega-1))];
m=zeros(1,n_omega);
for i=1:n_omega
    w=omega(i);
    if w==0
        rhs=(1-rate*tref)*Ps-P0;
        Psol=(-L-R)\rhs;
    else
        eref=exp(-1i*w*tref);
        Op=sparse(1:N^2,1:N^2,1i*w,N^2,N^2)-L-eref*R;
        rhs=(eref+rate*1i/w*(1-eref))*Ps-P0;
        Psol=Op\rhs;
    end
    m(i)=2*real(sum(R*Psol)*dv*da);
end
s=rate*(1+m);
f=omega/2/pi;

%% Plot
loglog(f,s,'LineWidth',2);
xlabel('Frequency (Hz)'); ylabel('S(f)');
title('Spike-train power spectrum (1D FPE, fixed kron form)');
grid on; toc;



function x_ref=gen_x_ref(v,t,p)
    if t==0
        x_ref=eye(p.N);
    else
        x_ref=zeros(p.N,p.N);
        Kappa=sqrt(p.tau_m/(pi*p.sigma^2*(1-exp(-2*t/p.tau_m))))*p.dv;
        for i=1:p.N
            x_ref(i,:)=Kappa*exp(-p.tau_m*(v(i)-p.I-exp(-t/p.tau_m)*(v(:)-p.I)).^2./(p.sigma^2)./(1-exp(-2*t/p.tau_m)));
        end
    end
end
    % row=zeros(N,N,2);col=zeros(N,N,2);value=zeros(N,N,2);
%     if vtag==1
%         for j=1:N
%             for k=1:N
%                 row(j,k,1)=v1_loc(j)+v2_loc(nr); col(j,k,1)=v1_loc(k)+v2_loc(nr); value(j,k,1)=x_ref(j,k);
%                 row(j,k,2)=v1_loc(j)+v2_loc(nr+1); col(j,k,2)=v1_loc(k)+v2_loc(nr+1); value(j,k,2)=x_ref(j,k);
%             end
%         end
%     elseif vtag==2
%         for j=1:N
%             for k=1:N
%                 row(j,k,1)=nr+v2_loc(j); col(j,k,1)=nr+v2_loc(k); value(j,k,1)=x_ref(j,k);
%                 row(j,k,2)=nr+1+v2_loc(j); col(j,k,2)=nr+1+v2_loc(k); value(j,k,2)=x_ref(j,k);
%             end
%         end
%     end
% row=reshape(row,1,[]); col=reshape(col,1,[]); value=reshape(value,1,[]); 
% x_reff=sparse(row,col,value,N^2,N^2);
