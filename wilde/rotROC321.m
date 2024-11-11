function R = rotROC321(a, b)
R = [1 0 -sin(b);
    0 cos(a) cos(b)*sin(a);
    0 -sin(a) cos(a)*cos(b)];
end