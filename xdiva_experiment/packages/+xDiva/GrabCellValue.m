function rV = GrabCellValue( cellArray, col1string, outputCol )
    % xDiva function by Spero to
    % pull out the column-2 (or other column) value of cell array row
    % where column-1 matches input string
    % example: xDiva.GrabCellValue( cellArray, col1string, outputCol );
    if nargin < 3
        outputCol = 2;
    end
    rV = cellArray{ strcmp( cellArray(:,1), col1string ), outputCol };
end