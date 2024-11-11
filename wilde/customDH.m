function T=customDH(a, theta)
%2D homogeneous transformation matrix using DH parameters

T_xA=[1 0 0 a;
    0 1 0 0;
    0 0 1 0;
    0 0 0 1];

T_zTheta=[cos(theta) -sin(theta) 0 0;
    sin(theta) cos(theta) 0 0;
    0 0 1 0;
    0 0 0 1];

T=T_zTheta*T_xA;
end