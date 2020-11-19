"""
 < Description >

 [M,E0,Eiter] = DMRG_1site (Hs,Nkeep,Nsweep [, option])

 Single-site density-matrix renormalization group (DMRG) calculation to
 search for the ground state and its energy of one-dimensional system,
 whose Hamiltonian is given by the matrix product operator Hs.

 < Input >
 Hs : [1 x N cell array] Matrix product operator (MPO) of the Hamiltonian.
       Each Hs{n} acts on site n, and is rank-4 tensor. The order of legs
       of Hs{n} is left-bottom-right-top, where bottom (top) leg is to be
       contracted with bra (ket) tensor. The length N of Hs, i.e.,
       numel(Hs), determines the chain length.
 Nkeep : [numeric] Maximum bond dimension of the matrix product state
       (MPS) to consider.
 Nsweep : [numeric] Number of sweeps will be 2 * Nsweep, as there are
       Nsweep times of round trip (right -> left, left -> right).

 < Option >
 'Morth', .. : [1 x N cell array] Reference MPS that makes the search of
       the ground state to be constrained to the subspace orthogonal to
       'Morth'. For example, if one inputs 'Morth' as the ground-state
       MPS, then the resulting M will be the first excited state.
 'rand' : If given, the MPS is initialized by random tensors. If not
       (which is default), it is initialized by the result of the
       iterative diagonalization from the left.

 < Output >
 M : [1 x N cell array] The result MPS which is obtained variationally to
       have the minimum expectation value of Hamiltonian H (or with a
       contraint, if 'Morth' is set). It is in *left-canonical* form,
       since the last sweep is from left to right.
 E0 : [numeric] The energy of M.
 Eiter : [N x (2*Nsweep) numeric array] Each element Eiter(m,n) means the
       variational energy in the m-th iteration within the n-th sweep.
       Odd n is for right-to-left sweep and even n for left-to-right
       sweep. Note that the iteration index m matches with the site index
       for left-to-right sweep; the iteration m corresponds to the site
       (N+1-m) for right-to-left sweep.

 Written by S.Lee (May 28,2019)
 Updated by S.Lee (May 23,2020): Revised for SoSe2020.
"""
function DMRG_1site(Hs,Nkeep,Nsweep,x...)

tic = time();

# default value of optional input
Morth = [];
isrand = false; # true: initialize M with random numbers

# parsing option
if !isempty(x)
	if x[1] == "Morth"
    		Morth = x[2];
	elseif x[1] == "rand"
    		isrand = true;
	else
    		error("ERROR: check input!");
	end
end

## sanity check for input and option
N = length(Hs);

if N < 2
    error("ERROR: chain is too short.");
elseif !isempty(Morth) && (length(Morth) != N)
    error("ERROR: Reference MPS Morth has different lengths from that of Hs.");
end

for itN in 1:N
    if size(Hs[itN],2) != size(Hs[itN],4)
        error("ERROR: The second and fourth legs of Hs[$itN] have different dimensions.");
    end
    if !isempty(Morth) && (size(Hs[itN],4) != size(Morth[itN],2))
        error("ERROR: The second leg of Morth[$itN] does not have compatible size with the fourth leg of Hs[$itN].");
    end
end
##

# show message
if isempty(Morth)
	println("Single-site DMRG: search for the ground state.");
else
	println("Single-site DMRG: search for the lowest-energy state orthogonal to ''Morth''.")
end

println("# of sites = $(length(Hs)), Nkeep = $Nkeep, # of sweeps = $(Nsweep*2).")

## Initialize the ket vector
M = Vector{Union{Array{Float64,2},Array{Float64,3}}}(undef, N);
for i in 1:length(M)
	M[i] =  Array{Float64,3}(undef, 0,0,0);
end
if isrand
    # initialize with random MPS
    for itN in 1:N
        if itN == 1
            M[itN] = rand(1,size(Hs[itN],4),Nkeep); # left leg is dummy
        elseif itN == N
            M[itN] = rand(Nkeep,size(Hs[itN],4),1); # right leg is dummy
        else
            M[itN] = rand(Nkeep,size(Hs[itN],4),Nkeep);
        end
    end
    M[end] = reshape(M[end],(30,2));
else
    # Initilize with the result of the iterative diagonalization from the
    # left.
    # Here the input Hamiltonian is defined by MPO, so the Hamiltonian for
    # a part of the chain has an open leg to the right. To diagonalize the
    # Hamiltonian, we choose the index 1 for the right leg. It is
    # consistent with the convention by which we define the Hamiltonian for
    # the last chain site.

    Hzip = [1]; 			# MPO Hamiltonian contracted by MPS kets/bras, "zipped" from 
    Hzip = reshape(Hzip,(1,1,1))	# the left end; initialize as 1 for dummy leg
    for itN = (1:N)
        if itN == 1
            M[itN] = getIdentity([1],3,Hs[itN],4); # left leg is dummy
        else
            M[itN] = getIdentity(M[itN-1],3,Hs[itN],4);
        end
        Hzip = updateLeft(Hzip,3,M[itN],Hs[itN],4,M[itN]);
        # Hamiltonian to be used for the iterative diagonalization
        Hzip2 = reshape(Hzip[:,1,:],(size(Hzip,1),size(Hzip,3)));
        # here we chose the index 1 for the 2nd leg, following the
        # convention of defining the right-most tensor within the MPO
        # Hamiltonian.
        D,V = eigen((Hzip2+Hzip2')/2);
        ids = sortperm(D);
        V = V[:,ids];
        if itN < N
            V = V[:,(1:min(size(V,2),Nkeep))];
        else
            V = V[:,1];
	    V = reshape(V,(length(V),1));
        end
        M[itN] = contract(M[itN],3,[3],V,2,[1]);
        Hzip = contract(Hzip,3,[3],V,2,[1]);
        Hzip = contract(conj(V),2,[1],Hzip,3,[1]);
    end
end

M[end] = reshape(M[end],(size(M[end],1),size(M[end],2)));

M,flagS,flagdw = canonForm(M,0); # bring into right-canonical form
M,flagS,flagdw = canonForm(M,length(M)); # bring into left-canonical form

# ground-state energy for each iteration
Eiter = zeros(N,2*Nsweep);
# later, Eiter(end,end) will be taken as the final result E0

## Hamiltonian for the left/right parts of the chain
Hlr = Vector{Array{Float64,3}}(undef, N+2);
for i in 1:length(Hlr)
        Hlr[i] =  Array{Float64,3}(undef, 0,0,0);
end
# Hlr{1} and Hlr{end} are dummies; they will be kept empty. These dummies
# are introduced for convenience.

# Since M is in left-canonical form by now, Hlr{..} are the left parts of
# the Hamiltonian. That is, Hlr{n+1} is the left part of Hamiltonian which
# is obtained by contracting M(1:n) with Hs(1:n). (Note the index for Hlr
# is n+1, not n, since Hlr{1} is dummy.)
for itN = (1:N)
    if itN == 1
        # "remove" the left leg (1st leg) which is dummy by permuting to the last
	H2 = permutedims(Hs[itN],[2 3 4 1]);
	H2 = reshape(H2,(size(H2,1),size(H2,2),size(H2,3)));
	Hlr[itN+1] = updateLeft([],[],M[itN],H2,3,M[itN]);
    else
	Hlr[itN+1] = updateLeft(Hlr[itN],3,M[itN],Hs[itN],4,M[itN]);
    end
end

if isrand
    print_time();
    println(" | Initialize with random MPS. Energy = $(Hlr[N+1])");
else
    print_time();
    println(" | Initialize with iterative diagonalization. Energy = $(Hlr[N+1])");
end

if !isempty(Morth)
    # the left and right parts of the overlap between M and Morth
    Olr = Vector{Array{Float64,2}}(undef, N+2);
	for i in 1:length(Olr)
        	Olr[i] =  Array{Float64,2}(undef, 0,0);
	end
    # Olr{1} and Olr{end} are dummies; they will be kept empty. These
    # dummies are introduced for convenience.

    # Since M and Morth are in left-canonical form by now, Olr{..} are the
    # left parts of the overlap. That is, Olr{n+1} is the left part of
    # overlap which is obtained by contracting M(1:n) with Morth(1:n). (Note
    # the index for Hlr is n+1, not n, since Hlr{1} is dummy.)
    for itN in 1:N
        Olr[itN+1] = updateLeft(Olr[itN],2,M[itN],[],[],Morth[itN]);
    end

    print_time();
    println("Overlap with Morth = $(Olr[N+1])")
end


for itS in (1:Nsweep)
    # right -> left
   for itN in (N:-1:1)
       if isempty(Morth)
           # Use eigs_1site to obtain the variationally chosen ket tensor
           # Aeff and energy expectation value Eeff
           Aeff,Eeff = eigs_1site(Hlr[itN],Hs[itN],Hlr[itN+2],M[itN]);
           #Aeff = reshape(Aeff,(size(Aeff,1),size(Aeff,2)));
       else
           # Find Aorth which is the ket tensor obtained by the overlap
           # between M and Morth (i.e., contracting the whole M and Morth except
           # for M{itN})

           # vector whose orthogonal subspace is to be considered
           Aorth = Morth[itN];
           if !isempty(Olr[itN])
               Aorth = contract(Olr[itN],2,[2],Aorth,3,[1]);
           end
           if !isempty(Olr[itN+2])
               Aorth = contract(Aorth,3,[3],Olr[itN+2],2,[2]);
           end

           # Use the optional input of eigs_1site to find Aeff and Eeff
           Aeff,Eeff = eigs_1site(Hlr[itN],Hs[itN],Hlr[itN+2],M[itN],"Aorth",Aorth);
       end


       Eiter[N+1-itN,2*itS-1] = Eeff;

       # update M{itN} and M{itN-1} by using Aeff, via SVD
       # decompose Aeff
       UT,ST,M[itN] = svdTr(Aeff,3,[1],Nkeep,[]);
	STdiag = zeros(size(ST,1),size(ST,1));
	for i in 1:size(ST,1)
		STdiag[i,i] = ST[i];
	end
       # contract UT*STdiag with M{itN}, to update M{itN}
       if itN > 1
           M[itN-1] = contract(M[itN-1],3,[3],UT*STdiag,2,[1]);
       end
       # update the Hamiltonian in effective basis
	T = permutedims(M[itN],[3 2 1]); # permute left<->right, to make use of updateLeft
       if itN == N
           # "remove" the right leg (3rd leg) of Hs{itN}, which is dummy,
           # by permuting to the last; MATLAB automatically suppresses the
           # trailing singleton dimensions
	    H2 = permutedims(Hs[itN],[2 1 4 3]); # bottom-left-top (-right)
	    H2 = reshape(H2,(size(H2,1),size(H2,2),size(H2,3)));
	    Hlr[itN+1] = updateLeft([],[],T,H2,3,T);
       elseif itN == 1
           Hlr[itN+1] = Array{Float64,3}(undef, 0,0,0);
       else
           # permute left<->right for Hs{itN} as well, to make use of updateLeft
           H2 = permutedims(Hs[itN],[3 2 1 4]); # right-bottom-left-top
           Hlr[itN+1] = updateLeft(Hlr[itN+2],3,T,H2,4,T);
       end
       if !isempty(Morth)
           # update the overlap between the current MPS and the reference
           # MPS which gives orthogonality constraint
           Olr[itN+1] = updateLeft(Olr[itN+2],2,T,[],[],permutedims(Morth[itN],[3 2 1]));
       end
   end

    # display informaiton of the sweep
    print_time();
    println(" | Sweep # $(2*itS-1)/$(2*Nsweep) (right -> left) : Energy = $(Eiter[N,2*itS-1])");
    if !isempty(Morth)
	println("Overlap with Morth = $(Olr[2])");
    end

    # left -> right
    for itN in 1:N
        if isempty(Morth)
            Aeff,Eeff = eigs_1site(Hlr[itN],Hs[itN],Hlr[itN+2],M[itN]);
        else
            # vector whose orthogonal subspace is to be considered
            Aorth = Morth[itN];
            if !isempty(Olr[itN])
                Aorth = contract(Olr[itN],2,[2],Aorth,3,[1]);
            end
            if !isempty(Olr[itN+2])
                Aorth = contract(Aorth,3,[3],Olr[itN+2],2,[2]);
            end

            Aeff,Eeff = eigs_1site(Hlr[itN],Hs[itN],Hlr[itN+2],M[itN],"Aorth",Aorth);
        end

        Eiter[itN,2*itS] = Eeff;

        # update M{itN} and M{itN+1} by using Aeff, via SVD
        # decompose Aeff
        M[itN],ST,VT = svdTr(Aeff,3,[1,2],Nkeep,[]);
	STdiag = zeros(size(ST,1),size(ST,1));
        for i in 1:size(ST,1)
                STdiag[i,i] = ST[i];
        end
        # contract UT*STdiag with M[itN], to update M{itN}
        if itN < N
	    M[itN+1] = contract(STdiag*VT,2,[2],M[itN+1],3,[1]);
        end

        # update the Hamiltonian in effective basis
        if itN == 1
            # "remove" the left leg (1st leg) of Hs{itN}, which is dummy,
            # by permuting to the last; MATLAB automatically suppresses the
            # trailing singleton dimensions
	    H2 = permutedims(Hs[itN],[2 3 4 1]); # bottom-right-top (-left)
	    H2 = reshape(H2,(size(H2,1),size(H2,2),size(H2,3)));
	    Hlr[itN+1] = updateLeft([],[],M[itN],H2,3,M[itN]);
        elseif itN == N
            Hlr[itN+1] = Array{Float64,3}(undef, 0,0,0);
        else
	    Hlr[itN+1] = updateLeft(Hlr[itN],3,M[itN],Hs[itN],4,M[itN]);
        end

        if !isempty(Morth)
            # update the overlap
            Olr[itN+1] = updateLeft(Olr[itN],2,M[itN],[],[],Morth[itN]);
        end
    end

    # display informaiton of the sweep
    print_time();
    println(" | Sweep # $(2*itS)/$(2*Nsweep) (left -> right) : Energy = $(Eiter[N,2*itS])");
    if !isempty(Morth)
	println("Overlap with Morth = $(Olr[N+1])");
    end
end

E0 = Eiter[end,end]; # take the last value
toc = time() - tic;

if isempty(Morth)
	println("Elapsed time: $(toc) s | Ground-state energy = $(E0)");
else
	println("Elapsed time: $(toc) s | First excited state energy = $(E0)");
end
return M,E0,Eiter;

end
