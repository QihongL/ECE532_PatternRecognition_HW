for i = 1 : 10000
    rng(i);
    m = 3;
    n = 10;
    A = randi(100, m,n);
    
    ATA = A'* A;
    
    rank(ATA);
    
    temp = ATA + eye(n,n);
    
    
    counter = 0 ;
    if rank(temp) ~= n
        counter = counter + 1;
    end
end
counter