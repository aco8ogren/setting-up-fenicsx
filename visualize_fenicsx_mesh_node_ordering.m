clear; close all;

% To use this script, copy the fenicsx_nodal_coordinates from the VTU file
% output by ex_create_hexahedral_mesh.py

fenicsx_nodal_coordinates = [0 0 0 0.5 0 0 0 0.5 0 0.5 0.5 0 0 0 0.5 0.5 0 0.5 0 0.5 0.5 0.5 0.5 0.5 1 0 0 1 0.5 0 1 0 0.5 1 0.5 0.5 0 1 0 0.5 1 0 0 1 0.5 0.5 1 0.5 0 0 1 0.5 0 1 0 0.5 1 0.5 0.5 1 1 1 0 1 1 0.5 1 0 1 1 0.5 1 0 1 1 0.5 1 1 1 1 1];

fenicsx_nodal_coordinates = fenicsx_nodal_coordinates';
fenicsx_nodal_coordinates = reshape(fenicsx_nodal_coordinates,3,[])';

number_of_nodes = size(fenicsx_nodal_coordinates,1);

node_labels = cellfun(@(x) num2str(x),num2cell(1:number_of_nodes),'uniformoutput',false);
spacing = 0.1;

coords = {fenicsx_nodal_coordinates};
titles = {'fenicsx node ordering'};

fig = figure;
tlo = tiledlayout(1,numel(coords));

for i = 1:length(titles)
    ax = nexttile;
    scatter3(coords{i}(:,1),coords{i}(:,2),coords{i}(:,3),'k','filled')
    text(coords{i}(:,1)+spacing,coords{i}(:,2)+spacing,coords{i}(:,3)+spacing,node_labels)
    daspect([1 1 1])
    xlabel('x')
    ylabel('y')
    zlabel('z')
    title(titles{i})
    axis padded
end
