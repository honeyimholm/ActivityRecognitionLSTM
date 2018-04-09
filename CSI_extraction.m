LABEL = [0 1]; %define label # for dataset

[filename, pathname] = uigetfile('*.dat');
csi_trace = read_bf_file(fullfile(pathname, filename));

N = 19979;%length(csi_trace);

trace = zeros(N, 90);
%19979 ok entries for hallwaydat1
%9980 ok entries for hallwaydat2
for i = 1:N
    %if(is_empty(csi_trace{i, 1}))
    csi = abs(squeeze(csi_trace{i, 1}.csi)); %use this one for non-dB
    %csi = db(abs(squeeze(csi_trace{i, 1}.csi))); %use this one for dB
    trace(i, :) = [csi(1, :), csi(2, :), csi(3, :)];
end
csvwrite([filename(1:end-4) '.csv'], trace);

label_mtx = repmat(LABEL, N, 1);
csvwrite([filename(1:end-4) '_label.csv'], label_mtx);