function R=rotation321(phi, thetaa, psi)
R_x=[1 0 0;
    0 cos(phi) -sin(phi);
    0 sin(phi) cos(phi)];
R_y=[cos(thetaa) 0 sin(thetaa);
    0 1 0;
    -sin(thetaa) 0 cos(thetaa)];
R_z=[cos(psi) -sin(psi) 0;
    sin(psi) cos(psi) 0;
    0 0 1];
R=R_z*R_y*R_x;
end