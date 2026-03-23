function createDataViewerUI(data_dir)
    % Create the figure window
    fig = figure('Name', 'Number Input UI', 'Position', [100, 100, 400, 150]);

    % Set initial number value
    number_value = 1;

    % Create a text box for inputting the number
    number_edit = uicontrol('Style', 'edit', 'Position', [150, 60, 100, 30], 'String', num2str(number_value), 'Callback', @(src, event) update_number(src));

    % Create the "Up" button
    up_button = uicontrol('Style', 'pushbutton', 'Position', [260, 60, 40, 30], 'String', '↑', 'Callback', @(src, event) increment_value(1));

    % Create the "Down" button
    down_button = uicontrol('Style', 'pushbutton', 'Position', [90, 60, 40, 30], 'String', '↓', 'Callback', @(src, event) increment_value(-1));

    % Function to increment the value
    function increment_value(change)
        number_value = number_value + change;
        set(number_edit, 'String', num2str(number_value));
        % Run the viewData function whenever the value is updated
        viewData(data_dir, number_value);
    end

    % Function to update the value when typed in the text box
    function update_number(src)
        number_value = str2double(get(src, 'String'));
        % Ensure the value is a valid number
        if isnan(number_value)
            number_value = 1;  % Default value if input is not valid
            set(src, 'String', num2str(number_value));
        end
        % Run the viewData function whenever the value is updated
        viewData(data_dir, number_value);
    end
end
