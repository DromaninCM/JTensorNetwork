"""
 M,S,dw = canonForm(M,id [,Nkeep]);

 Obtain the canonical forms of MPS. It brings the tensors M{1}, ..., M{id}
 into the left-canonical form and the others M{id+1}, ..., M{end} into the
 right-canonical form.

 < Input >
 M : [cell array] MPS of length numel(M). Each cell element is a rank-3
       tensor, where the first, second, and third dimensions are
       associated with left, bottom (i.e., local), and right legs,
       respectively.
 id : [integer] Index for the bond connecting the tensors M{id} and
       M{id+1}. With respect to the bond, the tensors to the left
       (right) are brought into the left-(right-)canonical form. If id ==
       0, the whole MPS will be in the right-canonical form.

 < Option >
 Nkeep : [integer] Maximum bond dimension. That is, only Nkeep the
       singular values and their associated singular vectors are kept at
       each iteration.

 < Output >
 M : [cell array] Left-, right-, or bond-canonical form from input M,
       depending on id, as follows:
       * id == 0: right-canonical form
       * id == numel(M): left-canonical form
       * otherwise: bond-canonical form
 S : [column vector] Singular values at the bond between M{id} and M{id+1}. 
 dw : [column vector] Vector of length numel(M)-1. dw(n) means the
       discarded weight (i.e., the sum of the square of the singular  
       values that are discarded) at the bond between M{n} and M{n+1}.

 Written by S.Lee (Apr.30,2019)
 Julia version by D. Romanin (November 4, 2020)
"""
function canonForm(M,id,x...)

# default option values
Nkeep = Inf; # keep all

# parsing option
if !isempty(x)
    Nkeep = x[1];
end
#

# check the integrity of input
if (length(id) != 1) || (round(Int,id) !== id)
    error("ERROR: 2nd input ''id'' needs to be a single integer.");
elseif (id < 0) || (id > length(M))
    error("ERROR: 2nd input ''id'' needs to be in a range (0:length(M))");
elseif size(M[1],1) != 1
    error("ERROR: the first dimension (= left leg) of M[1] should be 1.");
elseif size(M[end],3) != 1
    error("ERROR: the third dimension (= right leg) of M[end] should be 1.");
elseif (Nkeep <= 1) || isnan(Nkeep)
    error("ERROR: Option ''Nkeep'' should be positive integer larger than 1.");
end
#

dw = zeros(length(M)-1,1); # discarded weights

# Bring the left part of MPS into the left-canonical form
for i = (1:id)
    # reshape M[i] and SVD
    T = M[i];
    T = reshape(T,(size(T,1)*size(T,2),size(T,3)));
    U,S,V = svd(T);

    # truncate singular values/vectors; keep up to Nkeep. Truncation at the
    # bond between M[i] and M[i+1] is performed later.
    if !isinf(Nkeep) && (i < id)
        nk = min(length(S),Nkeep); # actual number of singular values/vectors to keep
        dw[i] = dw[i] + sum(S[nk+1:end].^2); # discarded weights
        U = U[:,1:nk];
        V = V[:,1:nk];
        S = S[1:nk]; 
    end

    # reshape U into rank-3 tensor, and replace M[i] with it
    M[i] = reshape(U,(convert(Int,size(U,1)/size(M[i],2)),size(M[i],2),size(U,2)));
    if i < id
        # contract S and V' with M[i+1]
        M[i+1] = contract(Diagonal(S)*V',2,2,M[i+1],3,1);
    else
        # R1: tensor which is the leftover after transforming the left
        #   part. It will be contracted with the counterpart R2 which is
        #   the leftover after transforming the right part. Then R1*R2 will
        #   be SVD-ed and its left/right singular vectors will be
        #   contracted with the neighbouring M-tensors.
        R1 = Diagonal(S)*V';
    end
end

# In case of fully right-canonical form; the above for-loop is not executed
if id == 0
    R1 = 1;
end

# Bring the right part into the right-canonical form
for i = (length(M):-1:id+1)
    # reshape M[i] and SVD
    T = M[i];
    T = reshape(T,(size(T,1),size(T,2)*size(T,3)));
    U,S,V = svd(T);

    # truncate singular values/vectors; keep up to Nkeep. Truncation at the
    # bond between M{id} and M{id+1} is performed later.
    if !isinf(Nkeep) && (i > (id+1))
        nk = min(length(S),Nkeep); # actual number of singular values/vectors to keep
        dw[i-1] = dw[i-1] + sum(S[nk+1:end].^2); # discarded weights
        U = U[:,1:nk];
        V = V[:,1:nk];
        S = S[1:nk]; 
    end

    # reshape V' into rank-3 tensor, replace M{it} with it
    M[i] = reshape(V',(size(V,2),size(M[i],2),convert(Int,size(V,1)/size(M[i],2))));

    if i > (id+1)
        # contract U and S with M{it-1}
        M[i-1] = contract(M[i-1],3,3,U*Diagonal(S),2,1);
    else
        # R2: tensor which is the leftover after transforming the right
        #   part. See the description of R1 above.
        R2 = U*Diagonal(S);
    end
end

# In case of fully left-canonical form; the above for-loop is not executed
if id == length(M)
    R2 = 1;
end

# SVD of R1*R2, and contract the left/right singular vectors to the tensors
U,S,V = svd(R1*R2);

# truncate singular values/vectors; keep up to Nkeep. At the leftmost and
# rightmost legs (dummy legs), there should be no truncation, since they
# are already of size 1.
if !isinf(Nkeep) && (id > 0) && (id < length(M))
    nk = min(length(S),Nkeep); # actual number of singular values/vectors to keep
    dw[id] = dw[id] + sum(S[nk+1:end].^2); # discarded weights
    U = U[:,1:nk];
    V = V[:,1:nk];
    S = S[1:nk];
end

if id == 0 # fully right-canonical form
    # U is a single number which serves as the overall phase factor to the
    # total many-site state. So we can pass over U to V'.
    M[1] = contract(U*V',2,2,M[1],3,1);
elseif id == length(M) # fully left-canonical form
    # V' is a single number which serves as the overall phase factor to the
    # total many-site state. So we can pass over V' to U.
    M[end] = contract(M[end],3,3,U*V',2,1);
else
    M[id] = contract(M[id],3,3,U,2,1);
    M[id+1] = contract(V'[:,:],2,2,M[id+1],3,1);
end

return M,S,dw

end
