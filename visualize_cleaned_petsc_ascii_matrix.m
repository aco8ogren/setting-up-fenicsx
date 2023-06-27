clear; close all;

fn = 'ex_write_stiffness_matrix_cleaned.txt';

% T = readtable(fn,'NumHeaderLines',0);  % skips the first three rows of data

% A = table2array(T);

import_opts = delimitedTextImportOptions;
import_opts.Delimiter = ' ';
import_opts.VariableTypes = 'double';

A = readmatrix(fn,import_opts);

% A = cell2mat(A);

imagesc(A)
daspect([1 1 1])
colorbar