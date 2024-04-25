filename = "labels/vial33_labels";
load(filename + ".mat")

labelData = gTruth.LabelData;

[rows, cols] = size(gTruth.LabelDefinitions);
for idx = 1:rows
    cellName = gTruth.LabelDefinitions(idx, :).Name;
    cellNameToString = char(cellName);
    countOnes = nnz(gTruth.LabelData.(cellNameToString));
    disp(cellNameToString + ": " + countOnes);
end

[rows_timetable, cols_timetable] = size(labelData);
% check if there are any rows which do not have a label
disp("Checking labels ")
for idx2 = 1:rows_timetable
    tt_row = labelData(idx2, :);

    sum_labels_without_useless = tt_row.still + tt_row.wave + tt_row.nearDrops + ...
    tt_row.smallDrops + tt_row.drops + tt_row.foam;
    
    sum_labels = sum_labels_without_useless + tt_row.useless;
    
    max_one_label_set = sum_labels_without_useless < 2;
    min_one_label_set = sum_labels > 0;
    
    % if there is no label set 
    if not(min_one_label_set)
        fprintf("%ss: No label set \n", tt_row.Time);
    end
    
    % if more than one of the labels are set e.g wave and still which is
    % not valid
    if not(max_one_label_set)
        fprintf("%ss: Too many labels \n", tt_row.Time);
    end
    
end