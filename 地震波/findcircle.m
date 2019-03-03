function [centersStrong,radiiStrong]=findcircle(A)
% Display the original image
[m,n,~]=size(A);
% Find all the circles with radius >= 15 and radius <= 30
% better: rmax < 3*rmin and (rmax-rmin) < 100.0
[centers, radii, metric] = imfindcircles(A,[floor(min(m,n)/6) floor(min(m,n)/2)]);
 % Retain the five strongest circles according to metric values
centersStrong=centers(1,:);
radiiStrong= radii(1);
metricStrong= metric(1);
 % Draw the circle perimeter for the five strongest circles
viscircles(centersStrong, radiiStrong,'Color','r');
hold on,pcolor(A),shading interp;
 