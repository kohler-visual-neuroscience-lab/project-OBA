function parameters = ReplaceParam( parameters, replacePart, replaceParam, replaceVal )
    % xDiva function by Alexandra Yakovleva to replace/correct 
    % parameters
    % xDiva.ReplaceParam( parameters replacePart, replaceParam, replaceVal )
    tPartLSS = ismember( { 'S' 'B' '1' '2' }, {replacePart} );
    tParamLSS = ismember( parameters{ tPartLSS }(:,1), {replaceParam} );
    parameters{ tPartLSS }{ tParamLSS, 2 } = replaceVal;
end