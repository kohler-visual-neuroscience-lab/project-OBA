function validationMessages = AppendVMs(validationMessages,aStr)
    % xDiva function by Alexandra Yakovleva to append to 
    % validation messages
    % validationMessages = xDiva.AppendWMs(validationMessages,aStr)
    
    if isempty(validationMessages{1})
        validationMessages{1} = aStr;
    else
        validationMessages = cat(1,validationMessages,{aStr});
    end
end

