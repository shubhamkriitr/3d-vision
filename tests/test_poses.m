%% Why?
% This script is used to determine the format of the coordinate system
% of the ScanNet dataset. This evaluation allowed us to determine that:
% x_cam: points to the right of the image plane created by the camera
% y_cam: points downwards w.r.t. the image plane created by the camera
% z_cam: point into the image plane created by the camera
% Hence, out of the two presumed cases, Case 2 is the correct one.


%% USER TODO BEFORE RUN
% define path to 3d-vision
project_path = "/home/nico/code/ETHZ/3d-vision";

% define poses to be analyzed
poses = ["0000", "0032", "0064", "0096"];
scene_difficulty = "easy_scenes";
scene = "scene0025_01";
%poses = ["1312", "1344", "1376", "1408"];
%scene_difficulty = "easy_scenes";
%scene = "scene0599_00";

%% load Rs and ts (from relative poses)
% vector colors
colors = ["r", "g", "b", "c", "m", "y", "k"];

% x, y, z scales
x_scale = 1;
y_scale = 0.5;
z_scale = 0.25;


poses_paths = [];
images_paths = [];

for i = poses
    poses_paths = [poses_paths, fullfile(project_path, "resources", scene_difficulty, scene, "relative_pose", i + ".txt")];
end
for i = poses
    images_paths = [images_paths, fullfile(project_path, "resources", scene_difficulty, scene, "images", i + ".png")];
end

% load Rs and ts
Rs = {};
ts = {};
for file_path = poses_paths
    rel_pose = readmatrix(file_path);
    Rs{end+1} = rel_pose(:, 1:3);
    ts{end+1} = rel_pose(:, 4).';
end

%% Plot images
plot_images(images_paths, poses);

%% Case 1: R_wc / -t_c(wc)
C_w = {};  % C_w are the camera x,y,z coordinates in world coordinates
for i = 1:length(Rs)
    C_w{end+1} = Rs{i}.';
end
T_w = {};
for i = 1:length(ts)
    T_w{end+1} = - Rs{i} * ts{i}.';
end
plot_poses(C_w, T_w, 'Case 1: R_wc / t_w(wc)', x_scale, y_scale, z_scale, colors, poses)

%% Case 2: R_cw / t_w(wc)
C_w = {};  % C_w are the camera x,y,z coordinates in world coordinates
for i = 1:length(Rs)
    C_w{end+1} = Rs{i};
end
T_w = {};
for i = 1:length(ts)
    T_w{end+1} = ts{i};
end
plot_poses(C_w, T_w, 'Case 2: R_cw / t_w(wc)', x_scale, y_scale, z_scale, colors, poses)


%% Plotting Functions
function plot_poses(C_w, T_w, title, x_scale, y_scale, z_scale, colors, poses)
    figure('Name', title);
    for i = 1:length(C_w)
        quiver3(T_w{i}(1), T_w{i}(2), T_w{i}(3), C_w{i}(1, 1)*x_scale, C_w{i}(2, 1)*x_scale, C_w{i}(3, 1)*x_scale, colors(i));
        hold on
        quiver3(T_w{i}(1), T_w{i}(2), T_w{i}(3), C_w{i}(1, 2)*y_scale, C_w{i}(2, 2)*y_scale, C_w{i}(3, 2)*y_scale, colors(i));
        quiver3(T_w{i}(1), T_w{i}(2), T_w{i}(3), C_w{i}(1, 3)*z_scale, C_w{i}(2, 3)*z_scale, C_w{i}(3, 3)*z_scale, colors(i));
    end
    %quiver3([0 5], [0 5], [0 5], [0 0], [0 0], [0, 0])
    xlabel('x');
    ylabel('y');
    zlabel('z');
    axis equal
    
    % add custom legend
    h = zeros(length(C_w), 1);
    for i = 1:length(C_w)
        h(i) = plot(NaN, NaN, append('-', colors(i)));
    end
    legend(h, poses);
    
end

function plot_images(images_paths, poses)
    figure
    images = {};
    for i = 1:length(images_paths)
        h = subplot(2, ceil(length(images_paths) / 2), i);
        img = imread(images_paths{i});
        image(img, 'Parent', h);
        title(poses(i));
    end
    
end


