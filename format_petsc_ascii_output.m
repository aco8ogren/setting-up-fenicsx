clear; close all;

in_fn = "ex_write_stiffness_matrix.txt";

out_fn = "ex_write_stiffness_matrix_cleaned.txt";

in_fid = fopen(in_fn);
data = fscanf(in_fid,'%c');
fclose(in_fid);

data_inds = 1:min(length(data),10000);

disp('Original data')
fprintf(data(data_inds))

fprintf('\n')

line_break = repelem('=',1,40);
disp(line_break)

clean_data = regexprep(data,'(\d*,','');
clean_data = regexprep(clean_data,')','');
clean_data = regexprep(clean_data,'Mat Object: 1 MPI processes','');
clean_data = regexprep(clean_data,'type: seqaij','');
clean_data = regexprep(clean_data,'row 0:',''); % Get rid of row 0:, it doesn't need a new line character
clean_data = regexprep(clean_data,'row \d*:','\\n'); % All other row *: replace with new line character
clean_data = regexprep(clean_data,'\s*',' ');
clean_data = regexprep(clean_data,'\\n ','\\n'); % spaces should not follow new lines
clean_data = regexprep(clean_data,' \\n','\\n'); % spaces should not preceed new lines
clean_data = clean_data(2:end); % Get rid of leading space
clean_data = clean_data(1:end-1); % Get rid of trailing space

clean_data_inds = 1:min(length(clean_data),10000);

disp('Clean data')
fprintf(clean_data(clean_data_inds))

out_fid = fopen(out_fn,'wt');
fprintf(out_fid,clean_data);
fclose(out_fid);