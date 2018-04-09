clc; clear; close all;

[filename, pathname] = uigetfile('*.dat');
csi_trace = read_bf_file(fullfile(pathname, filename));

T0 = csi_trace{1}.timestamp_low;

CarrierIndex = 5;

for i = 1:size(csi_trace,1)
    if(~isempty(csi_trace{i}))
        csi = get_scaled_csi(csi_trace{i});
        csi_db = db(abs(squeeze(csi).'));
        csi_single_carrier(i) = csi_db(CarrierIndex);
        time_arr(i) = csi_trace{i}.timestamp_low-T0;
    end
end
figure(1);
scatter(time_arr, csi_single_carrier,'linewidth',2); grid on;
title(['CSI for carrier ', num2str(CarrierIndex), ', hallway1 data']);
xlabel('Time (\mus)');
xlim([0 3e7]);
ylabel('SNR [dB]');
set(gca,'fontsize',14);

% figure(2);
% for i = 1:1:16
%     csi = get_scaled_csi(csi_trace{i});
%     subplot(4,4,i); plot(abs(squeeze(csi).')); grid on;
%     title(sprintf('Packet #%d', i));
%     xlabel('Subcarrier index');
%     ylabel('SNR Amplitude');
%     %set(gca,'fontsize',14);
% end

% figure(3);
% for i = 1:1:16
%     csi = get_scaled_csi(csi_trace{i});
%     subplot(4,4,i); plot(angle(squeeze(csi).')./pi); grid on;
%     title(sprintf('Packet #%d', i));
%     xlabel('Subcarrier index');
%     ylabel('SNR Phase / \pi');
%     ylim([-1, 1]);
%     %set(gca,'fontsize',14);
% end