"""
 contract(A,rankA,idA,B,rankB,idB [,idC] ['v']) 

 Contract tensors A and B. The legs to be contracted are given by idA
 and idB.

 < Input >
 A, B : [numeric array] Tensors.
 rankA, rankB : [integer] Rank of tensors. Since MATLAB removes the last
       trailing singleton dimensions, it is necessary to set rankA and
       rankB not to miss the legs of size 1 (or bond dimension 1,
       equivalently).
 idA, idB : [integer vector] Indices for the legs of A and B to be
        contracted. The idA(n)-th leg of A and the idB(n)-th leg of B will
        be contracted, for all 1 <= n <= numel(idA). idA and idB should
        have the same number of elements. If they are both empty, C will
        be given by the direct product of A and B.
 
 < Option >
 idC : [integer array] To permute the resulting tensor after contraction,
       assign the permutation indices as idC. If the dummy legs are
       attached (see the description of C below), this permutation is
       applied *after* the attachment.
       (Default: no permutation)
 'v' : Show details. Set this option to see how the legs of C are related
       to the legs of A and B.

 < Output >
 C : [numeric array] Contraction of A and B. If idC is given, the
       contracted tensor is permuted accordingly. If the number of open
       legs are smaller than 2, the dummy legs are assigned to make the
       result array C be two-dimensional.

 Written by S.Lee (Apr.30,2017)
 Updated by S.Lee (Apr.25,2019): Revised code.
 Julia version by D. Romanin (Nov.01,2020)
"""
function contract(A,rankA,idA,B,rankB,idB,x...)

# Default values of options
idC = []; # permutation of the contracted tensor (default: no permutation)
oshow = false; # option to show details

# parsing options
while !isempty(x)
    if length(x)==2 && isa(x[1],Array) && isequal(x[2],'v')
	idC = x[1];
	oshow = true;
	x = ();
    elseif isa(x[1],Array)
        idC = x[1];
        x = ();
    elseif isequal(x[1],'v')
        oshow = true;
        x = ();
    else
        show(x);
        error("Unkown option.")
    end
end
#

# Start measuring time if 'v'
if oshow
	start = time()
end
#

# Check the integrity of input and option
Asz = collect(size(A)); Bsz = collect(size(B)); # size of tensors
if rankA < length(Asz)
    error("ERROR: Input rankA is smaller than the rank of other input matrix A");
elseif rankB < length(Bsz)
    error("ERROR: Input rankA is smaller than the rank of other input matrix A");
end

# append 1's, in case that the tensor is indeed high-rank but the trailing
# singleton dimensions (i.e. dimensions of size 1)
Asz = append!(Asz,ones(1,rankA-length(Asz)));
Bsz = append!(Bsz,ones(1,rankB-length(Bsz)));

if length(idA) != length(idB)
    error("ERROR: Different # of leg indices to contract for tensors A and B.");
end
if !isempty(idA) # sanity check of idA and idB
    # logical array to check that idA has unique elements
    oks = all(i->(1<=i<=rankA), idA);
    if length(idA)>1
	okt = all(y->y==idA[1],idA);
    else
	okt = false
    end
    if !oks || okt  # if elements are not in range (1:rankA) or are all unique
	error("ERROR: idA=$idA should consist of unique integers in the range (1:$rankA).");
    end
    # likewise for idB
    oks = all(i->(1<=i<=rankB), idB);
    if length(idB)>1
	okt = all(y->y==idB[1],idB);
    else
	okt = false
    end
    if !oks || okt  # if elements are not in range (1:rankB) or are all unique
	error("ERROR: idB=$idB should consist of unique integers in the range (1:$rankB).");
    end
    if !all(Asz[idA]==Bsz[idB])
	error("ERROR: Dimensions of A to be contracted $Asz[idA] do not match with those of B $Bsz[idB].");
    end
end

# check whether permutation option is correct
if !isempty(idC) && (length(idC) < (rankA + rankB - 2*length(idA)))
	error("ERROR: # of indices for the permutation after contraction $idC is different from the # of legs after contraction $(rankA + rankB - 2*length(idA))")
end
#


### Main computational part (start) ###
# indices of legs *not* to be contracted
idA2 = collect((1:rankA)); idA2 = setdiff(idA2,idA);
idB2 = collect((1:rankB)); idB2 = setdiff(idB2,idB);

# reshape tensors into matrices with "thick" legs
if !isempty(idA2)
	idA3 = [];
	idA3 = append!(append!(idA3,idA2),idA);
	A2 = reshape(permutedims(A,idA3),(prod(Asz[idA2]),prod(Asz[idA])));
else
	A2 = reshape(permutedims(A,idA),(prod(Asz[idA])));
end
if !isempty(idB2)
        idB3 = [];
        idB3 = append!(append!(idB3,idB),idB2);
	B2 = reshape(permutedims(B,idB3),(prod(Bsz[idB]),prod(Bsz[idB2])));
else
	B2 = reshape(permutedims(B,idB),(prod(Bsz[idB])));
end
C2 = A2*B2; # matrix multiplication

# size of C
if (length(idA2) + length(idB2)) > 1
    Cdim = append!(reshape(Asz[idA2],(length(Asz[idA2]))),reshape(Bsz[idB2],length(Bsz[idB2])));
else
    # place dummy legs x of singleton dimension when all the legs of A (or
    # B) are contracted with the legs of B (or A)
    Cdim = [1,1];
    if !isempty(idA2)
        Cdim[1] = Asz[idA2];
    end
    if !isempty(idB2)
        Cdim[2] = Bsz[idB2];
    end
end
Cdim = Tuple(Cdim)

# reshape matrix to tensor
C = reshape(C2,Cdim);

if !isempty(idC) # if permutation option is given
    C = permutedims(C,idC);
end
### Main computational part (end) ###

# Stop measuring time if 'v'
if oshow
    # display result
    # text to contain leg and size information
    Astr = Vector{Union{Nothing, Int64, Tuple}}(nothing, length(idA2));
    Bstr = Vector{Union{Nothing, Int64, Tuple}}(nothing, length(idB2));
    for i in 1:length(idA2)
	Astr[i] = ('A',idA2[i],Asz[idA2[i]]); # leg index (size)
    end
    for i in 1:length(idB2)
	Bstr[i] = ('B',idB2[i],Bsz[idB2[i]]); # leg index (size)
    end

    if (length(idA2) + length(idB2)) > 1
        Cstr = [Astr,Bstr];
    else
	Cstr = Vector{Union{Int64, Tuple}}(1,2);
        if !isempty(idA2)
            Cstr[1] = Astr;
        end
        if !isempty(idB2)
            Cstr[2] = Bstr;
        end
    end

    if !isempty(idC) # if permutation option is given
        Cstr = permute!(Cstr,reshape(idC,length(idC)));
    end

    f = 0;
    println("Result leg (size): ")
    for i in 1:length(Cstr)
	for j in 1:length(Cstr[i])
		k = j + f
		println("C$k = $(Cstr[i][j][1])$(Cstr[i][j][2]) ($(Cstr[i][j][3]))")
	end
	f += length(Cstr[i]); 
    end

    # count elapsed time
        stop = time()
	println("Elapsed time: $(stop-start) s")
end
#

return C
end
