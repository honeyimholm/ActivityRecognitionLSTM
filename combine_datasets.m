csv1 = csvread('hallway1_label.csv');
csv2 = csvread('hallway2_label.csv');
allCsv = [csv1;csv2]; % Concatenate vertically
csvwrite('hallway_labels_combined.csv', allCsv);
csv1 = csvread('hallway1.csv');
csv2 = csvread('hallway2.csv');
allCsv = [csv1;csv2]; % Concatenate vertically
csvwrite('hallway_data_combined.csv', allCsv);