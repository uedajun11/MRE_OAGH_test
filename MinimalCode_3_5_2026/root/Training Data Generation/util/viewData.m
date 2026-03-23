function viewData(data_dir, index)
    % Plot out Wave and Inversion data in datadir
    diinversion_dir = fullfile(data_dir, 'DIinversion');
    lfeinversion_dir = fullfile(data_dir, 'LFEinversion');
    
    % Load the mat files for DI and LFE inversion
    di_files = dir(fullfile(diinversion_dir, '*.mat'));
    lfe_files = dir(fullfile(lfeinversion_dir, '*.mat'));
    
    % Load the data for the current index
    di_data = load(fullfile(diinversion_dir, di_files(index).name));
    lfe_data = load(fullfile(lfeinversion_dir, lfe_files(index).name));
    
    % Assume 'wave', 'mu', 'mfre', 'index', 'fov', 'sm_DI' are the variables
    wave = di_data.wave;  % Replace with the actual variable if needed
    mu = di_data.mu;      % Replace with the actual variable if needed
    DI = di_data.sm_DI;   % Replace with the actual variable if needed
    LFE = lfe_data.sm_LFE; % Replace with the actual variable if needed
    
    % Define the figure and check if it already exists
    fig_handle = findobj('Type', 'figure', 'Name', 'Inversion Data');
    
    if isempty(fig_handle)
        % Create a new figure if it doesn't exist
        fig_handle = figure('Name', 'Inversion Data');
    else
        % Use the existing figure
        figure(fig_handle);
    end

    cmap = jet;  % Choose a colormap for the wave images
    
    % Plot DI, LFE, and mu in the first row of subplots
    subplot(1, 3, 1), imshow(DI, [0 15]), title('DI'), colormap("jet"), colorbar;
    subplot(1, 3, 2), imshow(LFE, [0 15]), title('LFE'), colormap("jet"), colorbar;
    subplot(1, 3, 3), imshow(mu / 1000, [0 15]), title('True Mu'), colormap("jet"), colorbar;
    
    % Plot the 8 wave images in the second row of subplots
    c1 = [0 0 1]; 
    c2 = [0 0 0]; 
    c3 = [1 0 0]; 
    n = 256; 
    wave_cmap = interp1([1 n/2 n], [c1; c2; c3], 1:n);

    wave_fig_handle = findobj('Type', 'figure', 'Name', 'Wave Data');
    
    if isempty(wave_fig_handle)
        % Create a new figure if it doesn't exist
        wave_fig_handle = figure('Name', 'Wave Data');
    else
        % Use the existing figure
        figure(wave_fig_handle);
    end
    for i = 1:8
        subplot(2,4,i), imshow(wave(:,:,i), [-9e-4 9e-4]), colormap(wave_cmap), colorbar;
    end
end