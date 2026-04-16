function mae = computeMAE(image1, image2)
    % This function computes the Mean Absolute Error (MAE) between two images
    % Input: 
    %   image1 - the first image
    %   image2 - the second image (must have the same dimensions as image1)
    %
    % Output:
    %   mae - the Mean Absolute Error value

    % Ensure the images are the same size
    if size(image1) ~= size(image2)
        error('Input images must have the same size.');
    end

    % Compute the absolute difference between the two images
    diff = abs(image1 - image2);

    % Compute the mean of the absolute differences
    mae = mean(diff(:));
end