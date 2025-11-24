clear all; clc;

theta_b = 0:1:359;

mask =  [ %FR  %FL  %BR  %BL
        [ 100, 100, 100, 100]%F
        [   0, 100, 100,   0]%FR
        [-100, 100, 100,-100]%R
        [-100,   0,   0,-100]%BR
        [-100,-100,-100,-100]%B
        [   0,-100,-100,   0]%BL
        [ 100,-100,-100, 100]%L
        [ 100,   0,   0, 100]%FL
                            ];

for i = 0:1:359
    section = i/45;
    section_f = floor(section)+1;
    section_low = mask(section_f,:);
    section_high = mask(mod(section_f,8)+1,:);

    
    FR(i+1) = section_low(1) + (section_high(1) - section_low(1))*mod(i,45)/45;
    FL(i+1) = section_low(2) + (section_high(2) - section_low(2))*mod(i,45)/45;
    BR(i+1) = section_low(3) + (section_high(3) - section_low(3))*mod(i,45)/45;
    BL(i+1) = section_low(4) + (section_high(4) - section_low(4))*mod(i,45)/45;

end



figure;
subplot(4,1,1);
plot(theta_b, FR);
xlabel('Angle (Degrees)');
ylabel('Throttle (%)');
title('Front Right');
subplot(4,1,2);
plot(theta_b, FL);
xlabel('Angle (Degrees)');
ylabel('Throttle (%)');
title('Front Left');
subplot(4,1,3);
plot(theta_b, BR);
xlabel('Angle (Degrees)');
ylabel('Throttle (%)');
title('Back Right');
subplot(4,1,4);
plot(theta_b, BL);
xlabel('Angle (Degrees)');
ylabel('Throttle (%)');
title('Back Left');

