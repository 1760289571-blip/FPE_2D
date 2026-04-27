function results = matlab_validate_2d_psd(varargin)
% matlab_validate_2d_psd
% Simulate ONE 2D system (v,g) and compute PSD of firing events.
% Parameter defaults are aligned with 2_3Tensor_2D_neuron_v_g_spectrum.py.
%
% Example:
%   results = matlab_validate_2d_psd('omega',10,'T',200,'dt',1e-4);

p = inputParser;

% Simulation controls
p.addParameter('dt', 1e-4, @(x)isnumeric(x) && isscalar(x) && x>0);
p.addParameter('T', 200, @(x)isnumeric(x) && isscalar(x) && x>0);
p.addParameter('burn_in', 20, @(x)isnumeric(x) && isscalar(x) && x>=0);
p.addParameter('seed', 123, @(x)isnumeric(x) && isscalar(x));

% Model parameters (consistent names/defaults with theoretical python code)
p.addParameter('mu_ex', 8.0, @isscalar);
p.addParameter('mu_ffd', 0.0, @isscalar);
p.addParameter('tau_m', 0.02, @isscalar);
p.addParameter('tau_ee', 0.004, @isscalar);
p.addParameter('sigma_ex', 2.0, @isscalar);
p.addParameter('sigma_ffd', 0.8, @isscalar);
p.addParameter('p_ee', 0.2, @isscalar);
p.addParameter('s_ee', 1.0, @isscalar);
p.addParameter('V_thres', 20.0, @isscalar);
p.addParameter('V_reset', 0.0, @isscalar);
p.addParameter('v0', 0.0, @isscalar);
p.addParameter('g0', 0.0, @isscalar);

% PSD settings
p.addParameter('omega', 10.0, @(x)isnumeric(x) && isscalar(x) && x>=0); % rad/time
p.addParameter('nfft', 2^14, @(x)isnumeric(x) && isscalar(x) && x>0);
p.addParameter('plot_on', true, @(x)islogical(x) || isnumeric(x));
p.addParameter('save_mat', 'sec24_single_traj_validation.mat', @ischar);

p.parse(varargin{:});
P = p.Results;
rng(P.seed);

N_total = floor(P.T / P.dt);
N_burn = floor(P.burn_in / P.dt);
if N_burn >= N_total
    error('burn_in must be smaller than T.');
end

v = P.v0;
g = P.g0;

sigma_v_step = (P.sigma_ex / P.tau_m) * sqrt(P.dt);
sigma_g_step = (P.sigma_ffd * P.s_ee * sqrt(P.p_ee) / P.tau_ee) * sqrt(P.dt);

spike_times = zeros(1, max(128, floor((P.T-P.burn_in)/P.dt/20)));
n_spk = 0;

for k = 1:N_total
    v = v + ((-v + P.mu_ex + g) / P.tau_m) * P.dt + sigma_v_step * randn;
    g = g + ((-g + P.s_ee * P.mu_ffd * P.p_ee) / P.tau_ee) * P.dt + sigma_g_step * randn;

    if v >= P.V_thres
        if k > N_burn
            n_spk = n_spk + 1;
            if n_spk > numel(spike_times)
                spike_times = [spike_times, zeros(1, numel(spike_times))]; %#ok<AGROW>
            end
            spike_times(n_spk) = (k-1) * P.dt;
        end
        v = P.V_reset;
    end
end
spike_times = spike_times(1:n_spk);

T_eff = P.T - P.burn_in;
r0 = n_spk / T_eff;

% PSD at given omega from point-process Fourier sum
if isempty(spike_times)
    Fomega = 0;
else
    Fomega = sum(exp(-1i * P.omega * spike_times));
end
S_omega = (1 / T_eff) * abs(Fomega)^2;

% Optional diagnostic PSD curve from binned spike train
spk = zeros(1, N_total - N_burn);
if n_spk > 0
    idx = floor((spike_times - P.burn_in) / P.dt) + 1;
    idx = idx(idx >= 1 & idx <= numel(spk));
    spk(idx) = spk(idx) + 1;
end
x = spk - mean(spk);
X = fft(x, P.nfft);
Fs = 1 / P.dt;
f = (0:P.nfft-1) * (Fs / P.nfft);
omega_grid = 2*pi*f;
Pxx = (P.dt / numel(x)) * abs(X).^2;
half = floor(P.nfft/2) + 1;

results = struct();
results.params = P;
results.r0 = r0;
results.n_spikes = n_spk;
results.spike_times = spike_times;
results.omega = P.omega;
results.S_omega = S_omega;
results.omega_grid = omega_grid(1:half);
results.psd_curve = Pxx(1:half);

save(P.save_mat, '-struct', 'results');
fprintf('Saved %s\n', P.save_mat);
fprintf('r0=%.6g, spikes=%d, S(omega=%.6g)=%.6g\n', r0, n_spk, P.omega, S_omega);

if P.plot_on
    figure('Color','w');
    plot(results.omega_grid, results.psd_curve, 'k-', 'LineWidth',1.0); hold on;
    xline(P.omega, 'm--', 'target \omega');
    xlabel('\omega (rad/time)'); ylabel('PSD');
    title(sprintf('Spike-train PSD (single 2D trajectory), S(\\omega)=%.3g', S_omega));
    grid on;
end

end
