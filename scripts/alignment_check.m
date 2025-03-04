function alignment_check(directory_path)
    % Input validation
    if nargin < 1
        error('Please provide a directory path as an argument');
    end
    
    % Check if directory exists
    if ~exist(directory_path, 'dir')
        error('Directory does not exist: %s', directory_path);
    end
    
    % Change to the specified directory
    cd(directory_path);
    
    % Check if required files exist
    if ~exist('measurement.hdr', 'file') || ~exist('measurement.raw', 'file')
        error('measurement.hdr or measurement.raw not found in directory: %s', directory_path);
    end
    
    % Read the data
    info = enviinfo('measurement.hdr');
    hcube = hypercube(info.Filename);
    
    data = multibandread('measurement.raw', [info.Height, info.Width, info.Bands],...
        info.DataType, info.HeaderOffset, info.Interleave, info.ByteOrder);
    
    % Display the viewer
    hyperspectralViewer(hcube);
end