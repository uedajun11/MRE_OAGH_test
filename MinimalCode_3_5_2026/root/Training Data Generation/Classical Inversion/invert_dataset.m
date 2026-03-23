function invert_dataset(data_filepath)
    % INVERT_DATASET: Inverts the data and saves the results with new file 
    %                 names in ..._DIinversion and ... _LFEinversion.
    % 
    % Input:
    %   data_filepath - Directory containing the saved `.mat` files



    % Get all the files in the directory with a `.mat` extension
    files = dir(fullfile(data_filepath, '*.mat'));
    DI_dir = fullfile(data_filepath,'DIinversion');
    LFE_dir = fullfile(data_filepath, 'LFEinversion');

    if ~exist(DI_dir, 'dir')
        % Create the directory if it doesn't exist
        mkdir(DI_dir);
    end

    if ~exist(LFE_dir, 'dir')
        % Create the directory if it doesn't exist
        mkdir(LFE_dir);
    end
    
    for i = 1:length(files)
        % Load the dataset file
        file_name = files(i).name;
        data = load(fullfile(data_filepath, file_name));
        
        % Extract variables from the loaded data
        wave = data.wave;  % The wave field (y-displacements)
        mu = data.mu;      % Shear modulus (material property)
        mfre = data.mfre;  % Mechanical frequency
        index = data.index;  % Index of the data
        fov = data.fov;    % Field of view
        
        % Normalize wave 
        max_wave = max(max(max(wave)));
        min_wave = min(min(min(wave)));
        norm_wave = (wave-min_wave)./(max_wave-min_wave);

        % Apply inversion logic for DI (e.g., scaling or transforming data)
        sm_DI = Direct_Inverse(norm_wave,1000,fov,mfre,1000,3);

        % Apply inversion logic for LFE (e.g., scaling or filtering data)
        sm_LFE = MRE_LFE(norm_wave,6,6,[fov,fov],3,mfre,"noncomp"); % changed from 1, mfre to 3,mfre
        
        % Construct new file names
        [~, base_name, ext] = fileparts(file_name);
        output_file_DI = fullfile(DI_dir, [base_name, '_DIinversion', ext]);
        output_file_LFE = fullfile(LFE_dir, [base_name, '_LFEinversion', ext]);
        
        % Save the inversions as new .mat files
        save(output_file_DI, 'wave', 'mu', 'mfre', 'index', 'fov', 'sm_DI');
        save(output_file_LFE, 'wave', 'mu', 'mfre', 'index', 'fov', 'sm_LFE');
    end
end

