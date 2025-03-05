% helpful links I have used:
% https://www.mathworks.com/help/images/ref/enviinfo.html?s_tid=doc_ta
% https://www.mathworks.com/help/images/hyperspectral-image-processing.html
% https://www.mathworks.com/matlabcentral/answers/853385-reading-a-raw-file-from-a-hyperspectral-camera-using-envi-information
% https://www.mathworks.com/matlabcentral/answers/1807290-hyperspectral-imaging-signature-plot
% https://www.mathworks.com/help/images/ref/hyperspectralviewer-app.html

function alignment_check(directory_path)
    % Input validation
    if nargin < 1
        error('Please provide a directory path as an argument');
    end
    
    % Check if directory exists
    if ~exist(directory_path, 'dir')
        error('Directory does not exist: %s', directory_path);
    end
    
    % Create full file paths
    hdr_file = fullfile(directory_path, 'measurement.hdr');
    raw_file = fullfile(directory_path, 'measurement.raw');
    
    % Check if required files exist
    if ~exist(hdr_file, 'file') || ~exist(raw_file, 'file')
        error('measurement.hdr or measurement.raw not found in directory: %s', directory_path);
    end
    
    % Read the data
    info = enviinfo(hdr_file);
    hcube = hypercube(info.Filename);
    
    data = multibandread(raw_file, [info.Height, info.Width, info.Bands],...
        info.DataType, info.HeaderOffset, info.Interleave, info.ByteOrder);
    
    % Display the viewer
    hyperspectralViewer(hcube);
end