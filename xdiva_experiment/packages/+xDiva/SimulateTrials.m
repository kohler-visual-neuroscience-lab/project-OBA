function report = SimulateTrials( matlabFunction,parameters, timing, videoMode, numIter, loopVals )
    % xDiva function by PJ Kohler to simulate 1000s of trials, with a specified
    % set of parameters, for error checking purposes
    %
    % required input 
    %   - matlabFunction: name of xDiva MatlabFunction, as a string 
    %       (must be on path)
    %   - parameters: parameters cell, must match the specified function
    %   - timing:     timing cell, must match the specified function
    %   - videoMode:  videoMode cell, must match the specified function
    % optional input
    %   - numIter: number of iterations (default: 1000)
    %   - loopVals: 1 x 3 cell where the first element is the part string { 'S' 'B' '1' '2' }
    %               second element is the parameter name as a string 
    %               and third element is a cell of strings with parameter values to loop over
    % output
    %   - report: struct with report of simulated trials
    
    if nargin < 6
        nLoop = 1;
    else
        nLoop = length(loopVals{3});
    end
    if nargin < 5
        numIter = 1000;
    elseif nargin < 4
        error('both matlabfunction name, parameters, timing and videoMode are required');
    else
    end
    
%     if nLoop > 1 % if you are looping over a parameter
%         partList = { 'S' 'B' '1' '2' };
%         for q = 1:length(partList)
%             if ismember( parameters{ partList{q} }(:,1), iterParam );
%                 iterPart = partList{q};
%                 break; % leave the loop
%             else
%             end
%         end
%     end
        
    report = struct([]);
    for p = 1:nLoop
        if nLoop > 1 % if you are looping over parameter values
            parameters = xDiva.ReplaceParam( parameters, loopVals{1}, loopVals{2}, loopVals{3}{p} );
        else
        end
        fh = str2func(matlabFunction);
        report(p).runTime = zeros(1,numIter);
        report(p).parameters = parameters;
        for z = 1:numIter 
            tic;
            fh('MakeMovie',parameters,timing,videoMode,z);
            report(p).runTime(z) = toc;
            fprintf('Iteration # %d\n',z);
        end
    end
end