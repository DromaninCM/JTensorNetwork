"""
 < Description >

 [Lambda,Gamma,Es] = iTEBD_GS (Lambda,Gamma,H,Nkeep,betas [, option])

 Run iTEBD (infinite time-evolving block decimation) method to find the
 ground state, by using the imaginary time evolution. Here we consider
 only the unit cell of two sites.

 < Input >
 Lambda : [cell] Each cell contains the column vector of the singular
       values at each bond. Number of cells, numel(Lambda), means the size
       of the unit cell. numel(Lambda) should be 2.
 Gamma : [cell] Each cell contains a rank-3 tensor associated with each
       site within an unit cell. Number of cells, numel(Gamma), means the
       size of the unit cell, and needs to be the same as numel(Lambda) =
       2.
       The ket tensor for the unit cell is represented by:

 ->-diag(Lambda{2})->-*->-Gamma{1}->-*->-diag(Lambda{1})->-*->-Gamma{2}->-*->-diag(Lambda{2})->- 
  1                 2   1    ^     3   1                 2   1     ^    3   1                 2 
                             |2                                    |2

       The Lambda's and Gamma's can be given as random tensors. After the
       imaginary time evoltion, the contracted tensors Lambda{2}*Gamma{1}
       and Lambda{1}*Gamma{2} become left-normalized, and
       Gamma{1}*Lambda{1} and Gamma{2}*Lambda{2} become right-normalized,
       up to some numerical error. (Here we omitted diag(...) for Lambda's
       for brevity)
 H : [tensor] Two-site interaction Hamiltonian.
       Its leg convention is as below:

    2         4        [ 2 (4) is to be contracted with the second leg
    ^         ^          of Gamma{1} (Gamma{2}) ]
    |   ...   |
   [     H     ]
    |   ...   |
    ^         ^
    1         3

 Nkeep : [integer] Maximum bond dimension.
 betas : [numeric] Vector of imaginary time steps. 

 < Option >
 'Hastings' : Use the Hastings' method. In this case, the cell array Gamma
       represents "A" tensors which are supposed to be left-normalized.
       (Of course, they are not strictly left-normalized. This issue is
       explained in the demonstration and the exercise 3 in the tutorial
       T07.) Note that the Hastings' paper [M. Hastings, J. Math. Phys.
       50, 095207 (2009)] used the B tensors that are supposed to be
       right-normalized. Here we choose A tensors for convenience.
       The ket tensor for the unit cell is represneted by:

 ->-Gamma{1}->-*->-Gamma{2}->-*->-diag(Lambda{2})->-
  1    ^     3   1    ^     3   1                 2
       |2             |2 

       Note that there are no singular value tensors (Lambda's) sandwiched
       between Gamma's, contrary to Vidal's original Gamma-Lambda
       notation. Each Gamma{n} here is equivalent to
       diag(Lambda{mod(it,n)+1}) * Gamma{n} in Vidal's notation.

 < Output >
 Lambda, Gamma : [cell] Cell arrays of Lambda and Gamma tensors,
       repectively, after the imaginary time evolution. 
 Es : [numel(betas) x 2 matrix] Es(m,n) is the measured energy for the
       bond associated with singular values Lambda{n}, at the m-th step.

 Written by S.Lee (Jun.18,2017)
 Updated by S.Lee (Jun.19,2017)
 Updated by S.Lee (Jun.04,2019): Revised for Sose 2019.
 Julia Version by D.Romanin (Nov. 23,2020)
"""
function iTEBD_GS(Lambda,Gamma,H,Nkeep,betas,x...)

tic = time();

# default value of option
isHastings = false;

# parse option
if !isempty(x)
	if x[1] == "Hastings"
            isHastings = true;
        else
            error("ERROR: Unknown option.")
    	end
end
###

Nstep = length(betas);
Lambda = Lambda[:];
Gamma = Gamma[:];
n = length(Lambda); # number of sites in the unit cell
ldim = size(H,1); # local space dimension

### check the integrity of input
if n != 2
    error("ERROR: # of sites per unit cell should be 2.");
end

if ndims(H) > 4
    error("ERROR: H should be rank-4.");
elseif !any(i->(i in [size(H,2),size(H,3),size(H,4)]), ldim)
    error("ERROR: All the legs of H should have the same size.");
end

for it in (1:n)
    if length(Lambda[it]) != size(Gamma[it],3)
        error("ERROR: Dimensions for Lambda[$(it)] and Gamma[$(it)] do not match.");
    elseif length(Lambda[mod(it,n)+1]) != size(Gamma[it],1)
        error("ERROR: Dimensions for Lambda[$(mod(it,n)+1)] and Gamma[$(it)] do not match.");
    elseif size(Gamma[it],2) != ldim
        error("ERROR: The second leg of Gamma[$(it)] should be of size equal to the leg of H.");
    end
end
### 

# show information
print("iTEBD ground state search: ");
if !isHastings
    println("Vidal's method.");
else
    println("Hastings' method.");
end
println("# of sites = $(n), Bond dim. Nkeep = $(Nkeep), # of imag. time steps = $(Nstep)")

# energy expectation value at each step
Es = zeros(Nstep,n);

# diagonalize the Hamiltonian to exponentiate
Hmat = reshape(permutedims(H,[1,3,2,4]),(ldim^2,ldim^2)); # matrix representation
DH,VH = eigen((Hmat+Hmat')/2);

for it1 in 1:Nstep
    # exponentiate the matrix representation of Hamiltonian
    BH = zeros(length(DH),length(DH))
    for i in 1:length(DH)
	BH[i,i] = exp(-betas[it1]*DH[i]);
    end
    expH = VH*BH*VH';

    # reshape matrix -> rank-4 tensor
    expH = reshape(expH,(ldim,ldim,ldim,ldim));


    DL2 = zeros(length(Lambda[2]),length(Lambda[2]));
    DL1 = zeros(length(Lambda[1]),length(Lambda[1]));
    for i in 1:length(Lambda[2])
	DL2[i,i] = Lambda[2][i];
    end
    for i in 1:length(Lambda[1])
	DL1[i,i] = Lambda[1][i];
    end

    if !isHastings
        # T = Lambda{2}*Gamma{1}*Lambda{1}*Gamma{2}
        T = contract(DL2,2,[2],Gamma[1],3,[1]);
        T = contract(T,3,[3],DL1,2,[1]);
        T = contract(T,3,[3],Gamma[2],3,[1]);
    else
        # T = Gamma*Gamma
        T = contract(Gamma[1],3,[3],Gamma[2],3,[1]);
    end

    # contract exp(-betas(it1)*H) with ket tensors
    eHT = contract(expH,4,[3 4],T,4,[2 3],[3 1 2 4]);

    # contract Lambda{2} to the right
    Ttot = contract(eHT,4,[4],DL2,2,[1]);

    # do SVD: use svdTr
    UT,ST,VT = svdTr(Ttot,4,[1,2],Nkeep,[]);

    # normalize the singular value vector (so that the norm is 1)
    ST = ST/norm(ST);

    # update Lambda{1}
    Lambda[1] = ST;

    if !isHastings
        # update Gamma{1}, Gamma{2}
	IDL2 = zeros(length(Lambda[2]),length(Lambda[2]));
	for i in 1:length(Lambda[2])
		IDL2[i,i] = 1.0/Lambda[2][i];
	end
        Gamma[1] = contract(IDL2,2,[2],UT,3,[1]);
        Gamma[2] = contract(VT,3,[3],IDL2,2,[1]);

        # measure energy, for the bra/ket states of:
        # Lambda{2}*Gamma{1}*Lambda{1}*Gamma{2}*Lambda{2}*Gamma{1}*Lambda{1}
        #  ----------------   -------   ----------------
        #      = UT            = ST           = VT

        # energy at the bond for Lambda[1] (the 3rd tensor in the above form)
	DST = zeros(length(ST),length(ST));
	for i in 1:length(ST)
		DST[i,i] = ST[i];
	end
        US  = contract(UT,3,[3],DST,2,[1]);
        USV = contract(US,3,[3],VT,3,[1]);
        USV = reshape(USV,(size(USV,1),ldim^2,size(USV,4)));
        H2  = updateLeft([],[],USV,Hmat,2,USV);
        DL1 = zeros(length(Lambda[1]),length(Lambda[1]));
        for i in 1:length(Lambda[1])
                DL1[i,i] = Lambda[1][i];
        end
        GL  = contract(Gamma[1],3,[3],DL1,2,[1]);
        Es[it1,mod(it1-1,2)+1] = tr(updateLeft(H2,2,GL,[],[],GL));

        # energy at the bond for Lambda[2] (the 5th tensor in the above form)
        H2 = updateLeft([],[],US,[],[],US);
        VGL = contract(VT,3,[3],GL,3,[1]);
        VGL = reshape(VGL,(size(VGL,1),ldim^2,size(VGL,4)));
        Es[it1,mod(it1,2)+1] = tr(updateLeft(H2,2,VGL,Hmat,2,VGL));

        # normalize by the norm of the bra/ket states
        T = updateLeft([],[],US,[],[],US);
        T = updateLeft(T,2,VT,[],[],VT);
        T = updateLeft(T,2,GL,[],[],GL);
        Es[it1,:] = Es[it1,:]./tr(T);
    else
        # identify Gamma{1} as the unitary matrix of left singular vectors from svdTr
        Gamma[1] = UT;

        # obtain Gamma{2} by contracting Gamma{1} with the contraction  exp(-betas(it1)*H) * (Gamma*Gamma)
        Gamma[2] = contract(conj(Gamma[1]),3,[1 2],eHT,4,[1 2]);

        # normalize Gamma[1] by dividing a prefactor; it's crucial!
        G2G2 = contract(conj(Gamma[2]),3,[1 2],Gamma[2],3,[1 2]);
        Gamma[2] = Gamma[2]./sqrt(tr(G2G2)/size(G2G2,1));

        # measure energy, for the bra/ket states of:
        # Gamma{1}*Gamma{2}*Gamma{1}*Lambda{1}
        # (Note that here Gamma's are A tensors that are supposed to be left-normalized)

        # energy at the bond between the 1st and 2nd tensors in the above form
        G1G2 = contract(Gamma[1],3,[3],Gamma[2],3,[1]);
        G1G2 = reshape(G1G2,(size(G1G2,1),ldim^2,size(G1G2,4)));
        H2 = updateLeft([],[],G1G2,Hmat,2,G1G2);
	DL1 = zeros(length(Lambda[1]),length(Lambda[1]));
        for i in 1:length(Lambda[1])
                DL1[i,i] = Lambda[1][i];
        end
        G1L = contract(Gamma[1],3,[3],DL1,2,[1]);
        Es[it1,mod(it1-1,2)+1] = tr(updateLeft(H2,2,G1L,[],[],G1L));

        # energy at the bond between the 2nd and 3rd tensors in the above form
        G1G1 = updateLeft([],[],Gamma[1],[],[],Gamma[1]);
        GGL = contract(Gamma[2],3,[3],G1L,3,[1]);
        GGL = reshape(GGL,(size(GGL,1),ldim^2,size(GGL,4)));
        Es[it1,mod(it1,2)+1] = tr(updateLeft(G1G1,2,GGL,Hmat,2,GGL));

        # normalize by the norm of the bra/ket states
        T = updateLeft(G1G1,2,Gamma[2],[],[],Gamma[2]);
        T = updateLeft(T,2,G1L,[],[],G1L);
        Es[it1,:] = Es[it1,:]./tr(T);
    end

    # change the order of tensors to target the next bond
    Lambda = [Lambda[2],Lambda[1]];
    Gamma = [Gamma[2],Gamma[1]];

    if (mod(it1,500) == 0) && (it1 < Nstep)
	print_time();
	println(" | # $(it1)/$(Nstep), E = $(mean(Es[it1,:]))");
    end
end

print_time();
println(" | # $(Nstep)/$(Nstep), E = $(mean(Es[Nstep,:]))")

# check performance
toc = time()-tic;
println("Elapsed time = $(toc) s");

return Lambda,Gamma,Es;

end
